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

def createModelConfig():
    return util.loadConfig("model647m.json")


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
        assert dimFFN % dimEmb == 0, "sanity check"
        # similiar to wQKV, merge wGate and wValue into one larger linear
        self.wGateValue = torch.nn.Linear(dimEmb, dimFFN * 2, bias=False)
        self.wOut = torch.nn.Linear(dimFFN, dimEmb, bias=False)
        std = 0.02 / math.sqrt(2 * numLayer)
        torch.nn.init.normal_(self.wOut.weight, mean=0.0, std=std)
        self.dropout = torch.nn.Dropout(config["dropoutRate"])

    def forward(self, x):
        # SwiGLU(x) = (SiLU(x @ wGate) * x @ wValue) @ wOut
        # SiLU(x @ wGate) computes the 0~1 gate value to control how much
        # features from (x @ wValue) should be extracted and wOut projects
        # weighted features to real knowledge
        # Since (x @ wGate) and (x @ wValue) computes the same input, they can
        # be merged into one linear computation
        gv = self.wGateValue(x)
        wGate, wValue = gv.chunk(2, dim=-1)
        x = torch.nn.functional.silu(wGate) * wValue
        return self.wOut(self.dropout(x))


class Attention(torch.nn.Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        numLayer = config["numLayer"]
        self.dimEmb = config["dimEmb"]
        self.numHead = config["numHead"]
        assert self.dimEmb % self.numHead == 0, "sanity check"
        self.dropoutRate = config["dropoutRate"]
        # Use Kaiming initialization(Linear by default) for better convergence
        # merge separate Q,K,V into one large [Q,K,V] tensor for better efficiency
        self.wQKV = torch.nn.Linear(self.dimEmb, self.dimEmb * 3, bias=False)
        self.wOut = torch.nn.Linear(self.dimEmb, self.dimEmb, bias=False)
        std = 0.02 / math.sqrt(2 * numLayer)
        torch.nn.init.normal_(self.wQKV.weight, mean=0.0, std=std)
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
        # compute Q,K,V at once, and split them into appropriate Q,K,V, they are
        # all in shape of [batchSize, dimEmb, dimEmb], while wQKV is in shape of
        # [batchSize, dimEmb, 3 * dimEmb]
        qkv = self.wQKV(x)
        query, key, value = qkv.chunk(3, dim=-1)
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
    def __init__(self, config, tokenizer, device):
        super().__init__()
        torch.manual_seed(0xCAFEBABE)
        torch.set_float32_matmul_precision("high")
        dimEmb = config["dimEmb"]
        dimHead = dimEmb // config["numHead"]
        self.config = config
        self.tokenizer = tokenizer
        self.endOfTextId, _ = tokenizer.endOfText()
        self.vocabSize = tokenizer.vocabSize()
        self.device = device
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
    
    def accuracy(self, output, target, ignoreIndex):
        # find the index of the predicted token that has the highest probability
        valid = target != ignoreIndex
        predicted = output.argmax(dim=-1)
        # compare the predicted token with the target token
        correct = (predicted == target) & valid
        if valid.sum() > 0:
            return correct.sum() / valid.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def loss(self, output, target):
        # cross-entrypy loss asks for (numSample, numClass) and (numSample) as input
        # it means every sample has a prob distribution over all classes as output
        # and a single class as target
        # while I have out(batchSize, inputLen(numSample), vocabSize(numClass))
        # and target(batchSize, inputLen(numSample)), so I need to flatten them
        # as out(batchSize * inputLen, vocabSize) and target(batchSize * inputLen)
        output = output.view(output.shape[0] * output.shape[1], output.shape[2])
        target = target.view(target.shape[0] * target.shape[1])
        ignoreIndex = -100
        loss = torch.nn.functional.cross_entropy(output, target, ignore_index=ignoreIndex)
        # compute token accuracy = correct tokens / total tokens
        accu = self.accuracy(output, target, ignoreIndex)
        return loss, accu

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

class ModelOptim:
    def __init__(self, config, model, totalSteps):
        self.config = config
        self.model = model
        self.optimizer = self.setupOptimizer()
        self.scheduler = self.setupScheduler(totalSteps)
        self.gradNorm = None

    def setupOptimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.config["peakLR"], fused=util.cudaAvailable()
        )

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
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmupScheduler, decayScheduler],
            milestones=[warmupSteps],
        )

    def update(self):
        # prevent the exploding gradient problem
        self.gradNorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update learning rate
        self.scheduler.step()


class Scholar:
    """Scholar is used for training the model"""
    def __init__(self, config):
        self.config = config
        self.gradAccumStep = config["gradAccumStep"]
        self.tokenizer = tokenizer.createTokenizer(config["tokenizer"])
        # model and optimizer
        self.localRank, self.globalRank, self.worldSize = self.setupDDP()
        self.device = util.getTorchDevice(self.localRank)
        self.model = None
        self.modelOptim = None
        # dataset
        self.trainDataset = None
        self.valDataset = None
        self.trainSampler = None
        self.valSampler = None
        self.trainLoader = None
        self.valLoader = None
        # statistics
        self.bestValLoss = float("inf")
        self.stepsElapsed = 0
    
    def log(self, kvals):
        # only log on the rank-0 process
        util.log(kvals) if self.globalRank == 0 else None
    
    def saveModel(self, path):
        if self.globalRank != 0:
            return
        # model is wrapped at this point, so I need to access the original model
        # I need to save the best val loss otherwise the model will always update
        # this value when resuming training.
        torch.save(
            {
                "model": self.model.module.state_dict(),
                "optimizer": self.modelOptim.optimizer.state_dict(),
                "scheduler": self.modelOptim.scheduler.state_dict(),
                "bestValLoss": self.bestValLoss,
            },
            path,
        )
        self.log({"msg": f"Model saved to {path}"})
    
    def initModel(self, resumeData = None):
        # create model on specific device
        self.model = Model(self.config, self.tokenizer, self.device)
        self.model = (
            torch.compile(self.model) if sys.platform != "win32" else self.model
        )
        if resumeData != None:
            self.model.load_state_dict(resumeData["model"])
            self.bestValLoss = resumeData["bestValLoss"]

    def initModelOptim(self, epochSteps, resumeData = None):
        maxEpoch = self.config["numEpoch"]
        totalSteps = math.ceil(epochSteps * maxEpoch / self.gradAccumStep)
        self.modelOptim = ModelOptim(self.config, self.model, totalSteps)
        if resumeData != None:
            self.modelOptim.optimizer.load_state_dict(resumeData["optimizer"])
            self.modelOptim.scheduler.load_state_dict(resumeData["scheduler"])
    
    def initDataset(self):
        mixedDataset = dataloader.MixedDataLoader(self.config, self.tokenizer, self.worldSize)
        self.trainDataset = mixedDataset.getTrainDataset()
        self.valDataset = mixedDataset.getValDataset()
        self.trainSampler = torch.utils.data.distributed.DistributedSampler(
            self.trainDataset,
            num_replicas=self.worldSize,
            rank=self.globalRank,
            shuffle=True,
            seed=0xCAFEBABE,
        )
        self.trainLoader = torch.utils.data.DataLoader(
            self.trainDataset,
            sampler=self.trainSampler,
            batch_size=self.config["batchSize"],
            num_workers=1,
            pin_memory=True,
        )
        self.valSampler = torch.utils.data.distributed.DistributedSampler(
            self.valDataset,
            num_replicas=self.worldSize,
            rank=self.globalRank,
            shuffle=False,
            seed=0xCAFEBABE,
        )
        self.valLoader = torch.utils.data.DataLoader(
            self.valDataset,
            sampler=self.valSampler,
            batch_size=self.config["batchSize"],
            num_workers=4,
            pin_memory=True,
        )

    def setupDDP(self):
        # torchr automatically setups RANK/LOCAL_RANK/WORLD_SIZE env variables
        # RANK indicates the global rank of the process
        # LOCAL_RANK indicates the rank of the process on the current node
        # WORLD_SIZE indicates the total number of processes
        torch.distributed.init_process_group(group_name="gscholar",backend="nccl")
        localRank = int(os.getenv("LOCAL_RANK", "0"))
        globalRank = int(os.getenv("RANK", "0"))
        worldSize = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(localRank)
        return localRank, globalRank, worldSize
    
    def cleanupDDP(self):
        torch.distributed.destroy_process_group()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        lossAccum = torch.tensor(0.0, device=self.device)
        accuAccum = torch.tensor(0.0, device=self.device)
        numBytes = torch.tensor(0, dtype=torch.long, device=self.device)
        stepIdx = 0
        for batch in self.valLoader:
            stepIdx += 1
            inputs, targets = batch["inputs"], batch["targets"]
            for seq in inputs.tolist():
                seqBytes = self.tokenizer.decode(seq).encode('utf-8')
                numBytes += len(seqBytes)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            # compute loss without updating weights
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_bf16_supported(),
            ):
                output = self.model.forward(inputs)
                loss, accu = self.model.module.loss(output, targets)
            lossAccum += loss.detach()
            accuAccum += accu.detach()

        # reduce all replicas to get the average loss and accuracy
        torch.distributed.all_reduce(lossAccum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(accuAccum, op=torch.distributed.ReduceOp.SUM)
        # also reduce numBytes if we are in DDP
        torch.distributed.all_reduce(numBytes, op=torch.distributed.ReduceOp.SUM)

        # calculate the average loss and accuracy
        totalLoss = lossAccum.item() / (stepIdx * self.worldSize)
        totalAccu = accuAccum.item() / (stepIdx * self.worldSize)
        # calculate the bpb
        numTokens = self.config["maxWindowSize"] * self.config["batchSize"] * stepIdx * self.worldSize
        bpb = (totalLoss / math.log(2)) * numTokens / numBytes.item()
        return totalLoss, totalAccu, bpb

    def checkpoint(self, stepIdx, lossAccum, accuAccum):
        stepInterval = self.gradAccumStep * 60
        statistics = {}

        if stepIdx % stepInterval == 0:
            # On all replicas:
            # ReduceOp must be performed on all replicas
            torch.distributed.all_reduce(lossAccum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(accuAccum, op=torch.distributed.ReduceOp.SUM)
            trainLoss = (lossAccum / (stepInterval * self.worldSize)).item()
            trainAccuracy = (accuAccum / (stepInterval * self.worldSize)).item()
            stepsElapsed = time.time() - self.stepsElapsed
            throughput = self.config["maxWindowSize"] * self.config["batchSize"] * stepInterval * self.worldSize
            throughput = throughput / stepsElapsed
            gradNorm = self.modelOptim.gradNorm.item()
            currentLR = self.modelOptim.optimizer.param_groups[0]["lr"]
            statistics["event"] = "metrics"
            statistics["step"] = stepIdx
            statistics["currentLR"] = currentLR
            statistics["gradNorm"] = gradNorm
            statistics["trainLoss"] = trainLoss
            statistics["trainAccuracy"] = trainAccuracy
            statistics["throughpt"] = throughput

            # cleanup up
            lossAccum.zero_()
            accuAccum.zero_()
            self.stepsElapsed = time.time() # reset the timer
    
            if stepIdx % (stepInterval*2) == 0:
                # On all replicas:
                # validate the model performance
                valLoss, valAccuracy, valBPB = self.validate()
                statistics["valLoss"] = valLoss
                statistics["valPPL"] = math.exp(valLoss)
                statistics["valAccuracy"] = valAccuracy
                statistics["valBPB"] = valBPB
                # On rank-0 replica:
                # find best model
                if self.globalRank == 0:
                    if valLoss < self.bestValLoss:
                        self.saveModel("scholar_best.bin")
                        self.bestValLoss = valLoss
                    self.saveModel("scholar_last.bin")
            # On rank-0 replica:
            if self.globalRank == 0 and len(statistics) > 0:
                self.log(statistics)


    def bootTrain(self, resume):
        # setup the data loader
        self.initDataset()
        data = torch.load(resume, weights_only=False, map_location=self.device) if resume != "" else None
        self.initModel(data)
        epochSteps = len(self.trainLoader)
        self.initModelOptim(epochSteps, data)
        # wrap the model so that it can be trained in distributed manner
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.localRank])
        # log the training configuration
        totalParams = sum(p.numel() for p in self.model.module.parameters())
        self.log(
            {
                "event": "config",
                "worldSize": self.worldSize,
                "bf16": torch.cuda.is_bf16_supported(),
                "deviceCount": torch.cuda.device_count(),
                "flashAttn": torch.backends.cuda.flash_sdp_enabled(),
                "device": str(self.device),
                "vocabSize": self.tokenizer.vocabSize(),
                "params": totalParams,
                **self.config,
            }
        )
        self.log({"event": "startTrain", "timestamp": time.time()})

    def postTrain(self):
        self.cleanupDDP()
        self.log({"event": "endTrain", "timestamp": time.time()})

    def trainStep(self, input, target):
        self.model.train()
        input = input.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_bf16_supported(),
        ):
            output = self.model.forward(input)
            loss, accu = self.model.module.loss(output, target)
        # I update the weights every gradAccumStep steps, so the backward loss
        # should be scaled as well to compensate for the smaller update frequency
        lossScaled = loss / self.gradAccumStep
        lossScaled.backward()
        return loss, accu

    def train(self, resume=""):
        self.bootTrain(resume)
        stepIdx = 0
        lossAccum = torch.tensor(0.0, device=self.device)
        accuAccum = torch.tensor(0.0, device=self.device)
        print(f"@@ Training on rank {self.globalRank}/{self.worldSize}...")
        for epoch in range(self.config["numEpoch"]):
            epochStart = time.time()
            self.trainSampler.set_epoch(epoch)
            self.stepsElapsed = time.time()
            for batch in self.trainLoader:
                stepIdx += 1
                stepUpdate = stepIdx % self.gradAccumStep == 0
                inputs, targets = batch["inputs"], batch["targets"]
                # gradient sync is only performed at the end of each mini-batch
                if stepUpdate:
                    loss, accu = self.trainStep(inputs, targets)
                else:
                    with self.model.no_sync():
                        loss, accu = self.trainStep(inputs, targets)
                # accumulate loss and accuracy on GPU
                lossAccum += loss.detach()
                accuAccum += accu.detach()
                # update weights, logging, etc
                if stepUpdate:
                    self.modelOptim.update()
                    self.checkpoint(stepIdx, lossAccum, accuAccum)
            epochEnd = time.time()
            self.saveModel("scholar_last.bin")
            self.log({"epoch": epoch, "elapsed": epochEnd - epochStart})
        self.postTrain()


class Predictor:
    """Predictor is used for predicting the next token"""
    def __init__(self, config, path):
        self.config = config
        self.tokenizer = tokenizer.createTokenizer(config["tokenizer"])
        self.device = util.getTorchDevice(0)
        self.model = Model(config, self.tokenizer, self.device)
        data = torch.load(path, map_location=self.device)
        state_dict = data["model"]
        # Remove '_orig_mod.' prefix added by torch.compile if present
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)

    @torch.no_grad()
    def predict(self, sentence, numNextToken=50):
        self.model.eval()
        return self.model.nextToken(sentence, numNextToken)