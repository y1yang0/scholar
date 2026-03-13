# Copyright (c) 2026 yyang. All rights reserved.
import tokenizer
import dataloader
import util
import torch
import time
import sys
import os
import math
import json

IsDebug = False
Inspector = util.Inspector()

Sentences = [
    "杨过和小龙女在",
    "神雕大侠",
    "韦小宝和双儿",
    "围攻光明顶",
    "郭靖和黄蓉",
    "张无忌",
    "令狐冲说",
    "华山论剑",
    "桃花岛上",
    "令狐冲看到盈盈，问道",
    "他使出一剑",
    "女子冷哼一声",
    "你懂剑法吗？",
    "一剑刺中了",
    "东方不败冷笑一声，手中的绣花针",
    "乔峰大喝一声，掌力如怒潮般向慕容复",
    "张无忌回想起赵敏那盈盈笑语，又念及周芷若的楚楚可怜，一时间",
    "两人双掌相交，只听得“砰”的一声巨响，登时",
    "那长剑来势奇快，剑尖已堪堪触及他胸口大穴，就在这千钧一发之际，",
    "此时正值深夜，窗外狂风大作，暴雨倾盆。破庙之中，",
    "望着他远去的背影，她心中一阵酸楚，两行清泪",
    "老者厉声喝道：“你这贼子，今日休想活着离开！”说罢，",
]


def createModelConfig():
    config = {
        "dataset": [
            "data/small",
            "data/extend"
        ],
        "dimEmb": 640,
        "dimFFN": int(4 * 640),  # int(2 / 3 * 4 * 384)
        "numLayer": 16,
        "numHead": 10,
        "maxWindowSize": 768,
        "dropoutRate": 0.3,
        "peakLR": 3e-4,
        "minLR": 3e-5,
        "numEpoch": 3,
        "batchSize": 16,
        "trainDataRatio": 0.90,
        "temperature": 0.7,
        "topP": 0.85,
        "repeatPenalty": 1.3,
    }
    return config


class Normalization(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = torch.nn.RMSNorm(config["dimEmb"])

    def forward(self, x):
        return self.norm(x)


class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        dimEmb = config["dimEmb"]
        dimFFN = config["dimFFN"]
        numLayer = config["numLayer"]
        self.wGate = torch.nn.Linear(dimEmb, dimFFN, bias=False)
        self.wValue = torch.nn.Linear(dimEmb, dimFFN, bias=False)
        self.wOut = torch.nn.Linear(dimFFN, dimEmb, bias=False)
        std = 0.02 / math.sqrt(2 * numLayer)
        torch.nn.init.normal_(self.wOut.weight, mean=0.0, std=std)
        self.dropout = torch.nn.Dropout(config["dropoutRate"])

    def forward(self, x):
        # SwiGLU(x) = (SiLU(x @ wGate) * x @ wValue) @ wOut
        # SiLU(x @ wGate) computes the 0~1 gate value to control how much
        # features from (x @ wValue) should be extracted and wOut projects
        # weighted features to real knowledge
        x = torch.nn.functional.silu(self.wGate(x)) * self.wValue(x)
        return self.wOut(self.dropout(x))


class Attention(torch.nn.Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        dimEmb = config["dimEmb"]
        numLayer = config["numLayer"]
        self.numHead = config["numHead"]
        self.dropoutRate = config["dropoutRate"]
        # Use Kaiming initialization for better convergence
        self.wQuery = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wKey = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wValue = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wOut = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        std = 0.02 / math.sqrt(2 * numLayer)
        torch.nn.init.normal_(self.wOut.weight, mean=0.0, std=std)
        self.dropout = torch.nn.Dropout(self.dropoutRate)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def applyRoPE(self, q, k, inputLen):
        # q and k are (batchSize, numHead, inputLen, dimHead)
        # cos and sin are (inputLen, dimHead//2)
        cos, sin = self.cos[:inputLen, :], self.sin[:inputLen, :]
        # cos and sin are (1, 1, inputLen, dimHead//2)
        # now they are matched with Q and K
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        # qeven, qodd are (batchSize, numHead, inputLen, dimHead//2)
        # where last dimension is [q0,q2,q4...] [q1,q3,q5...]
        qeven, qodd = q[..., ::2], q[..., 1::2]
        keven, kodd = k[..., ::2], k[..., 1::2]
        # q0*cos(θ) - q1*sin(θ)
        # q1*cos(θ) + q0*sin(θ)
        # ... and so on
        rotatedQeven = qeven * cos - qodd * sin
        rotatedQodd = qodd * cos + qeven * sin
        rotatedKeven = keven * cos - kodd * sin
        rotatedKodd = kodd * cos + keven * sin
        # rotatedQ and rotatedK are (batchSize, numHead, inputLen, dimHead//2, 2)
        # so I should flatten the last dimension to get back to
        # (batchSize, numHead, inputLen, dimHead)
        rotatedQ = torch.stack([rotatedQeven, rotatedQodd], dim=-1).flatten(-2)
        rotatedK = torch.stack([rotatedKeven, rotatedKodd], dim=-1).flatten(-2)
        return rotatedQ, rotatedK

    def flashAttn(self, xshape, queries, keys, values):
        # While I enjoy coding from scratch and manually implementing key model
        # components, I realize that building a functional version of Flash
        # Attention is a massive undertaking. To accelerate the pre-training
        # process, I’ve decided to leverage torch's SPDA
        dropoutRate = self.dropoutRate if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=dropoutRate, is_causal=True
        )
        batchSize, inputLen, dimEmb = xshape
        out = out.transpose(1, 2).contiguous().view(batchSize, inputLen, dimEmb)
        return self.wOut(out)

    def forward(self, x):
        # compute Q,K,V at once, they are in shape of [batchSize, dimEmb, dimEmb]
        query = self.wQuery(x)
        key = self.wKey(x)
        value = self.wValue(x)
        # split the Q,K,V tensor into multiple heads, each head has dimHead
        # dimensions. Intuitively, I view old [batchSize, inputLen, dimEmb] as
        # [batchSize, numHead, inputLen, dimHead], but it turns out that it
        # should be firstly viewed as [batchSize, inputLen, numHead, dimHead]
        # and transpose(1,2) dimensions to get the desired shape
        batchSize, inputLen, dimEmb = x.shape
        dimHead = dimEmb // self.numHead
        queries = query.view(batchSize, inputLen, self.numHead, dimHead).transpose(1, 2)
        keys = key.view(batchSize, inputLen, self.numHead, dimHead).transpose(1, 2)
        values = value.view(batchSize, inputLen, self.numHead, dimHead).transpose(1, 2)
        # use RoPE to understand relative position of tokens
        queries, keys = self.applyRoPE(queries, keys, inputLen)
        # use torch's spda to leverage flash attention
        if not os.getenv("MISS_OLD_DAYS", "0") == "1":
            return self.flashAttn(x.shape, queries, keys, values)
        # compute Attention(Q,K,V) = softmax(mask(Q@K^T / sqrt(d_k))) @ V
        #
        # attention socre means which tokens are most relevant to current token
        #   Q(batchSize, numHead, inputLen, dimHead) @ K^T(batchSize, numHead, dimHead, inputLen)
        #   = attnScore(batchSize, numHead, inputLen, inputLen)
        attnScore = queries @ keys.transpose(-2, -1) / (dimHead**0.5)
        # use causal mask to prevent the current token from seeing future tokens
        #   attnScore(batchSize, numHead, inputLen, inputLen) @ mask(batchSize, numHead, inputLen, inputLen)
        #   = maskedAttnScore(batchSize, numHead, inputLen, inputLen)
        mask = torch.tril(torch.ones(inputLen, inputLen, device=x.device))
        attnScore = attnScore.masked_fill(mask == 0, -torch.inf)
        # apply softmax to get the attention weights
        attnWeights = torch.softmax(attnScore, dim=-1)
        # apply dropout to prevent overfitting
        attnWeights = self.dropout(attnWeights)
        # apply weights to the values to get the output
        #   attnWeights(batchSize, numHead, inputLen, inputLen) @ V(batchSize, numHead, inputLen, dimHead)
        #   = out(batchSize, numHead, inputLen, dimHead)
        out = attnWeights @ values
        # merge all attention heads back and apply final projection to understand
        # how to combine the information from all heads
        #   out(batchSize, numHead, inputLen, dimHead)
        #   = out(batchSize, inputLen, dimEmb)
        out = out.transpose(1, 2).contiguous().view(batchSize, inputLen, dimEmb)
        return self.wOut(out)


class Transformer(torch.nn.Module):
    def __init__(self, idx, config, cos, sin):
        super().__init__()
        self.idx = idx
        self.attn = Attention(config, cos, sin)
        self.norm1 = Normalization(config)
        self.norm2 = Normalization(config)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        Inspector.trace(f"Attn#{self.idx}", x) if IsDebug else None
        x = x + self.ffn(self.norm2(x))
        Inspector.trace(f"FFN#{self.idx}", x) if IsDebug else None
        return x


class Model(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        torch.manual_seed(0xCAFEBABE)
        torch.set_float32_matmul_precision("high")
        dimEmb = config["dimEmb"]
        dimHead = dimEmb // config["numHead"]
        self.config = config
        self.tokenizer = tokenizer
        self.endOfTextId, _ = tokenizer.endOfText()
        self.vocabSize = tokenizer.vocabSize()
        self.device = util.getTorchDevice()
        self.tokenEmbedding = torch.nn.Embedding(self.vocabSize, dimEmb)
        cos, sin = self.initRoPE(config["maxWindowSize"], dimHead)
        self.transformers = torch.nn.ModuleList(
            [Transformer(idx, config, cos, sin) for idx in range(config["numLayer"])]
        )
        self.finalNorm = Normalization(config)
        self.out = torch.nn.Linear(
            dimEmb, self.vocabSize, bias=False
        )  # no weight tying
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=config["peakLR"], fused=util.cudaAvailable()
        )
        self.scheduler = None
        self.gradNorm = None

    def setupScheduler(self, totalSteps):
        warmupSteps = int(totalSteps * 0.1)
        # start from peakLR * 0.01 to peakLR * 1
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmupSteps
        )
        decaySteps = int(totalSteps - warmupSteps)
        # decay from peakLR * 1 to minLR
        decayScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=decaySteps, eta_min=self.config["minLR"]
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmupScheduler, decayScheduler],
            milestones=[warmupSteps],
        )

    def initRoPE(self, maxWindowSize, dimHead):
        # freq = 10000 ^ (-2 * i / dimHead), where i is in [0, 1,..., dimHead//2]
        i = torch.arange(start=0, end=dimHead // 2, device=self.device)
        freq = 10000.0 ** (-2 * i / dimHead)
        pos = torch.arange(maxWindowSize, device=self.device)
        theta = torch.outer(pos, freq)
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        return cos, sin

    def forward(self, input):
        x = self.tokenEmbedding(input)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.finalNorm(x)
        Inspector.trace("FinalNorm", x) if IsDebug else None
        return self.out(x)

    def loss(self, output, target):
        # cross-entrypy loss asks for (numSample, numClass) and (numSample) as input
        # it means every sample has a prob distribution over all classes as output
        # and a single class as target
        # while I have out(batchSize, inputLen(numSample), vocabSize(numClass))
        # and target(batchSize, inputLen(numSample)), so I need to flatten them
        # as out(batchSize * inputLen, vocabSize) and target(batchSize * inputLen)
        output = output.view(output.shape[0] * output.shape[1], output.shape[2])
        target = target.view(target.shape[0] * target.shape[1])
        loss = torch.nn.functional.cross_entropy(output, target, ignore_index=-100)
        return loss

    def updateWeight(self):
        # prevent the exploding gradient problem
        self.gradNorm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update learning rate
        self.scheduler.step()

    def saveWeights(self, path):
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )
        util.log({"msg": f"Model saved to {path}"})

    def loadWeights(self, path):
        data = torch.load(path, weights_only=False, map_location=self.device)
        self.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.scheduler.load_state_dict(data["scheduler"]) if self.scheduler else None
        util.log({"msg": f"Model loaded from {path}"})

    def topP(self, logits):
        logits, idx = torch.sort(logits, descending=True)
        # [0.5,0.3,0.1,0.1]
        probs = torch.softmax(logits, dim=-1)
        # [0.5,0.8,0.9,1.0] if topP=0.85
        cum = torch.cumsum(probs, dim=-1)
        # [False, False, True, True]
        removeMask = cum > self.config["topP"]
        # keep the first token that makes cumulative probability exceed topP.
        # e.g., keep 0.1 so (0.5+0.3+0.1) >= 0.85
        removeMask[1:] = removeMask[:-1].clone()
        # keep at least one token in case of all tokens are removed
        removeMask[0] = False
        masked = logits.masked_fill(removeMask, -torch.inf)
        filtered = logits.clone()
        filtered.fill_(-torch.inf)
        filtered.scatter_(dim=-1, index=idx, src=masked)
        return filtered

    def repeatScale(self, logits, tokens):
        lookbackWindow = 20
        recentTokens = set(tokens[-lookbackWindow:])
        for rt in recentTokens:
            if logits[rt] > 0:
                logits[rt] = logits[rt] / self.config["repeatPenalty"]
            else:
                logits[rt] = logits[rt] * self.config["repeatPenalty"]

    @torch.no_grad()
    def nextToken(self, sentence, numNextToken=1):
        tokens = self.tokenizer.encode(sentence)
        # make sure the number of tokens is less than maxWindowSize
        # this is not strictly necessary because we have RoPE after all
        tokens = tokens[-self.config["maxWindowSize"] :]
        numInitTokens = len(tokens)
        for i in range(numNextToken):
            if i == numNextToken - 1:
                # only trace last token generation
                Inspector.start(self.tokenizer, tokens, self.out)
            t = torch.tensor(tokens, dtype=torch.long, device=self.device)
            logits = self.forward(torch.stack([t]))
            # first batch, last tokens, all logits
            logits = logits[0, -1, :]
            # find recently generated tokens and avoid repeating them
            self.repeatScale(logits, tokens)
            # creativity control
            logits = logits / self.config["temperature"]
            # cumulative prob of all candidates should exceed topP
            logits = self.topP(logits)
            probs = torch.softmax(logits, dim=-1)
            nextTokenId = torch.multinomial(probs, num_samples=1)
            # stop saying if we reach the end of text token
            if nextTokenId.item() == self.endOfTextId:
                break
            tokens.append(nextTokenId.item())
        return self.tokenizer.decode(tokens[numInitTokens:])


class Scholar:
    def __init__(self, config):
        self.config = config
        self.tokenizer = tokenizer.BBPETokenizer()
        self.model = Model(config, self.tokenizer)
        self.dataloader = None
        # training statistics
        self.totalBatch = 0
        self.bestValLoss = float("inf")

    def logConfig(self):
        totalParams = sum(p.numel() for p in self.model.parameters())
        util.log(
            {
                "event": "config",
                "bf16": torch.cuda.is_bf16_supported(),
                "flashAttn": torch.backends.cuda.flash_sdp_enabled(),
                "device": str(self.model.device),
                "vocabSize": self.tokenizer.vocabSize(),
                "params": totalParams,
                **self.config,
            }
        )

    def checkpoint(self, loss):
        self.totalBatch += 1
        if self.totalBatch % 50 == 0:
            currentLR = self.model.optimizer.param_groups[0]["lr"]
            util.log(
                {
                    "batch": self.totalBatch,
                    "trainLoss": loss.item(),
                    "gradNorm": self.model.gradNorm.item(),
                    "currentLR": currentLR,
                }
            )
        if self.totalBatch % 100 == 0:
            # do real text generation
            self.model.eval()
            for i, sentence in enumerate(Sentences):
                output = self.model.nextToken(sentence, numNextToken=20)
                util.log({"idx": i, "input": sentence, "output": output})
            self.model.train()
            # validate the model performance on validation set
            avgValLoss = self.validate()
            util.log(
                {
                    "batch": self.totalBatch,
                    "trainLoss": loss.item(),
                    "valLoss": avgValLoss,
                    "trainPPL": math.exp(loss.item()),
                    "valPPL": math.exp(avgValLoss),
                }
            )
            if avgValLoss < self.bestValLoss:
                self.model.saveWeights("scholar_best.bin")
                self.bestValLoss = avgValLoss
            self.model.saveWeights("scholar_last.bin")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        totalLoss = 0.0
        totalBatch = 0
        for input, target in self.dataloader.nextValBatch():
            input = input.to(self.model.device, non_blocking=True)
            target = target.to(self.model.device, non_blocking=True)
            # compute loss without updating weights
            with torch.autocast(
                device_type=self.model.device.type,
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_bf16_supported(),
            ):
                output = self.model.forward(input)
                loss = self.model.loss(output, target)
            totalLoss += loss.item()
            totalBatch += 1
        return totalLoss / totalBatch

    def bootTrain(self, resume):
        # use simple data loader to load Jinyong's novels all at once
        # it should be replaced with large data loader for streaming
        maxEpoch = self.config["numEpoch"]
        dataset = self.config["dataset"]
        files = util.listFiles(dataset, ".txt")
        # self.dataloader = dataloader.LargeDataLoader(self.config, self.tokenizer, files)
        self.dataloader = dataloader.SimpleDataLoader(
            self.config, self.tokenizer, files
        )
        self.model.setupScheduler(self.dataloader.totalTrainBatches() * maxEpoch)
        if resume != "":
            self.model.loadWeights(resume)
        self.model = (
            torch.compile(self.model) if sys.platform != "win32" else self.model
        )
        self.logConfig()

    def trainStep(self, input, target):
        self.model.train()
        input = input.to(self.model.device, non_blocking=True)
        target = target.to(self.model.device, non_blocking=True)
        with torch.autocast(
            device_type=self.model.device.type,
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_bf16_supported(),
        ):
            output = self.model.forward(input)
            loss = self.model.loss(output, target)
        loss.backward()
        return loss

    def train(self, resume=""):
        self.bootTrain(resume)
        print(f"@@ Training...")
        for epoch in range(self.config["numEpoch"]):
            epochStart = time.time()
            for input, target in self.dataloader.nextTrainBatch():
                loss = self.trainStep(input, target)
                self.model.updateWeight()
                self.checkpoint(loss)
            epochEnd = time.time()
            self.model.saveWeights("scholar_last.bin")
            util.log({"epoch": epoch, "elapsed": epochEnd - epochStart})

    def tuning(self):
        self.config["peakLR"] = 3e-5
        self.dataloader = dataloader.SFTDataLoader(self.config, self.tokenizer)
        self.train()

    @torch.no_grad()
    def predict(self, sentence, numNextToken=50):
        self.model.eval()
        self.model.loadWeights("scholar_last.bin")
        pairs = []
        output = self.model.nextToken(sentence, numNextToken)
        pairs.append((sentence, output))
        return pairs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        util.log({"event": "start", "mode": mode, "timestamp": time.time()})
        g = Scholar(createModelConfig())
        if mode == "train":
            resume = "" if len(sys.argv) <= 2 else sys.argv[2]
            g.train(resume)
            util.log({"event": "end", "timestamp": time.time()})
        elif mode == "predict":
            sentences = sys.argv[2]
            response = g.predict(sentences)
            json.dump(response, sys.stdout, ensure_ascii=False)
        elif mode == "tuning":
            g.tuning()
        elif mode == "debug":
            sentence = "过儿"
            IsDebug = True
            print(f"@@ DebugInput: {sentence}")
            [(_, output)] = g.predict([sentence], numNextToken=1)
            print(f"@@ DebugOutput: {output}")
    else:
        print("       python scholar.py train [resume.bin]")
        print('Usage: python scholar.py predict "杨过跳了下去，发现"')
        print("       python scholar.py debug")
