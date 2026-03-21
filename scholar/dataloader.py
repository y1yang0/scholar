# Copyright (c) 2026 yyang. All rights reserved.
import functools
import torch
import os
import random
import json
import util
import tokenizer
import datasets
import glob

class StreamDataLoader:
    def __init__(self, config, tokenizer, files):
        self.files = files
        self.maxWindowSize = config["maxWindowSize"]
        self.batchSize = config["batchSize"]
        self.tokenizer = tokenizer
        self.numTokens = 0
        self.fileIdx = 0
        self.filePos = 0
        self.endOfTextId, _ = tokenizer.endOfText()

    def packToBatch(self, chunks):
        # pack the dataset into smaller batches, i.e.
        # [(input, target), (input1, target1), ...] =>
        # [input, input1, ...], [target, target1, ...]
        assert len(chunks) == self.batchSize, "why not otherwise"
        random.shuffle(chunks)
        # [input, input1, ...], [target, target1, ...]
        inputBatch, targetBatch = zip(*chunks)
        # tensor([input, input1, ...]), tensor([target, target1, ...])
        return torch.stack(inputBatch), torch.stack(targetBatch)

    def read(self, f):
        return f.read(4096)

    def buildChunk(self, tokenIds):
        tokens = torch.tensor(tokenIds, dtype=torch.long)
        self.numTokens += tokens.numel()
        inputTokens = tokens[:-1]
        targetTokens = tokens[1:]
        return (inputTokens, targetTokens)

    def progress(self):
        return (self.fileIdx / len(self.files)) * 100

    def nextBatch(self):
        tokenBuf = []
        chunkBuf = []
        for i, file in enumerate(self.files):
            self.fileIdx = i
            size = os.path.getsize(file)
            print(f"@@ Loading... {i}/{len(self.files)}")
            with open(file, "r", encoding="utf-8") as f:
                while True:
                    t = self.read(f)
                    if not t:  # end of file, stop reading
                        tokenBuf.append(self.endOfTextId)
                        break
                    self.filePos = (f.tell() / size) * 100.0
                    tokenBuf.extend(self.tokenizer.encode(t))
                    # token buffer is still not full, keep reading
                    if len(tokenBuf) < (self.maxWindowSize + 1):
                        continue
                    # token buffer is full, split into many (input,target) chunks
                    i = 0
                    while (i + self.maxWindowSize + 1) <= len(tokenBuf):
                        # convert an aligned chunk of token ids to (input, target)
                        # tensor pair
                        tokenIds = tokenBuf[i : i + self.maxWindowSize + 1]
                        chunk = self.buildChunk(tokenIds)
                        chunkBuf.append(chunk)
                        # num of chunks are satisfied, pack to batch and yield
                        if len(chunkBuf) == self.batchSize:
                            inputBatch, targetBatch = self.packToBatch(chunkBuf)
                            chunkBuf = []
                            yield inputBatch, targetBatch
                        # caution: there should be a step of self.maxWindowSize
                        i += self.maxWindowSize
                    # shrink to unused tokens
                    tokenBuf = tokenBuf[i:]


class SimpleDataLoader:
    """
    Load all data at once, used for small dataset, e.g. Jinyong 15 novels
    """

    def __init__(self, config, tokenizer):
        files = util.listFiles(["data/small", "data/extend"], ".txt", [])
        loader = StreamDataLoader(config, tokenizer, files)
        batches = []
        self.numTokens = 0
        for inputBatch, targetBatch in loader.nextBatch():
            self.numTokens += inputBatch.numel()
            batches.append((inputBatch, targetBatch))
        random.shuffle(batches)
        trainDataRatio = config["trainDataRatio"]
        self.idx = 0
        self.trainBatches = batches[: int(len(batches) * trainDataRatio)]
        self.valBatches = batches[int(len(batches) * trainDataRatio) :]
        util.log(
            {
                "event": "config",
                "DatasetTokens": self.numTokens,
                "TrainBatches": len(self.trainBatches),
                "ValBatches": len(self.valBatches),
            }
        )

    def nextValBatch(self):
        random.shuffle(self.valBatches)
        for inputBatch, targetBatch in self.valBatches:
            self.idx += 1
            yield inputBatch, targetBatch

    def nextTrainBatch(self):
        random.shuffle(self.trainBatches)
        for inputBatch, targetBatch in self.trainBatches:
            self.idx += 1
            yield inputBatch, targetBatch

    def totalTrainBatches(self):
        return len(self.trainBatches)


class SFTDataLoader:
    """
    Load jsonl data, e.g. SFT dataset
    """

    def __init__(self, config, tokenizer, trainFiles, valFiles):
        self.endOfTextId, self.endOfText = tokenizer.endOfText()
        self.config = config
        self.trainFiles = trainFiles
        self.valFiles = valFiles

    def nextValBatch(self):
        for inputBatch, targetBatch in self.nextBatch(self.valFiles):
            yield inputBatch, targetBatch

    def nextTrainBatch(self):
        for inputBatch, targetBatch in self.nextBatch(self.trainFiles):
            yield inputBatch, targetBatch

    def nextBatch(self, files):
        chunkBuf = []
        for path in files:
            with open(path, "r", encoding="utf-8") as jsonl:
                while True:
                    line = jsonl.readline()
                    if not line:  # end of file, stop reading
                        break
                    data = json.loads(line)
                    question = f"问: {data['instruction']} 答:"
                    answer = f"{data['output']}{self.endOfText}"
                    questionIds = self.tokenizer.encode(question)
                    answerIds = self.tokenizer.encode(answer)
                    tokenIds = questionIds + answerIds
                    lenMaxWindow = self.maxWindowSize + 1

                    # drop this sft line if it's too long
                    if len(tokenIds) > lenMaxWindow:
                        continue
                    lenPad = lenMaxWindow - len(tokenIds)
                    # pad tokens to maxWindowSize
                    tokenIds.extend([self.endOfTextId] * lenPad)

                    # cross-entropy ignores -100, so mask question
                    tokens = torch.tensor(tokenIds, dtype=torch.long)
                    self.numTokens += tokens.numel()
                    # target = masked question + answer + padding
                    target = torch.tensor(
                        [-100] * len(questionIds) + answerIds + [-100] * lenPad,
                        dtype=torch.long,
                    )
                    chunkBuf.append((tokens[:-1], target[1:]))
                    if len(chunkBuf) == self.batchSize:
                        inputBatch, targetBatch = self.packToBatch(chunkBuf)
                        chunkBuf = []
                        yield inputBatch, targetBatch
        # sft data is precious, don't let it go to waste
        if len(chunkBuf) > 0:
            inputBatch, targetBatch = self.packToBatch(chunkBuf)
            yield inputBatch, targetBatch

def alignedChunk(tokenizer, maxWindowSize, fields, batch):
    tokenBuf = []
    numRows = len(batch[fields[0]])
    for rowIdx in range(numRows):
        texts = []
        for field in fields:
            value = batch[field][rowIdx]
            value = str(value).strip()
            if value:
                texts.append(value)
        if not texts:
            continue
        t = "".join(texts)
        endOfTextId, _ = tokenizer.endOfText()
        tokenBuf.extend(tokenizer.encode(t))
        tokenBuf.append(endOfTextId)

    inputs = []
    targets = []
    i = 0
    chunkSize = maxWindowSize + 1  # +1 for target
    while (i + chunkSize) <= len(tokenBuf):
        tokenIds = tokenBuf[i : i + chunkSize]
        inputs.append(tokenIds[:-1])
        targets.append(tokenIds[1:])
        i += maxWindowSize
    remains = len(tokenBuf) - i
    # I spend a lot of time collecting them, dont waste any tokens
    if remains > 1 and len(tokenBuf) >= chunkSize:
        lastWindowIdx = len(tokenBuf) - chunkSize
        inputIds = tokenBuf[lastWindowIdx:-1]
        targetIds = tokenBuf[lastWindowIdx + 1 :]
        inputs.append(inputIds)
        targets.append(targetIds)
    return {"inputs": inputs, "targets": targets}

class MixedDataLoader:
    def __init__(self, config, tokenizer, worldSize):
        self.config = config
        self.tokenizer = tokenizer
        self.worldSize = worldSize
        self.maxWindowSize = config["maxWindowSize"]
        self.batchSize = config["batchSize"]

        self.allDatasets = {
            "zhihu": self.addZhihuDataset(),
            "novel": self.addNovelDataset(),
            "wiki": self.addWikiDataset(),
            "liter": self.addLiterDataset(),
            "fineweb": self.addFinewebDataset(),
            "cqia": self.addCQIADataset(),
            "code": self.addCodeDataset(),
        }

        ds = [self.allDatasets[k] for k in self.allDatasets]
        self.mixedDataset = datasets.concatenate_datasets(ds)
        # self.mixedDataset = self.mixedDataset.shuffle(seed=0xCAFEBABE)
        self.mixedDataset.set_format(type="torch", columns=["inputs", "targets"])
        splitDataset = self.mixedDataset.train_test_split(
            test_size=0.1, seed=0xCAFEBABE, shuffle=False
        )
        self.trainDataset = splitDataset["train"]
        self.valDataset = splitDataset["test"]
        self.printStats()
    
    def addNovelDataset(self):
        novelDataset = datasets.load_dataset(
            "y1yang0/scholar-novels-curated", split="train"
        )
        novelDataset = novelDataset.map(
            functools.partial(alignedChunk, self.tokenizer, self.maxWindowSize, ["text"]),
            remove_columns=novelDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return novelDataset
    
    def addWikiDataset(self):
        wikiDataset = datasets.load_dataset(
            "shaowenchen/wiki_zh", split="train"
        )
        wikiDataset = wikiDataset.map(
            functools.partial(alignedChunk, self.tokenizer, self.maxWindowSize, ["text"]),
            remove_columns=wikiDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return wikiDataset
    
    def addLiterDataset(self):
        # 1g per file
        literDataset = datasets.load_dataset(
            "BAAI/IndustryCorpus_literature",
            data_files=[
                "zh/rank_1242.jsonl.gz",
                "zh/rank_1243.jsonl.gz",
                "zh/rank_1244.jsonl.gz",
                "zh/rank_1245.jsonl.gz",
                "zh/rank_1246.jsonl.gz",
                "zh/rank_1247.jsonl.gz",
            ],
            trust_remote_code=True,
            split = "train"
        )
        literDataset = literDataset.map(
            functools.partial(alignedChunk, self.tokenizer, self.maxWindowSize, ["text"]),
            remove_columns=literDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return literDataset
    
    def addFinewebDataset(self):
        # 9m per file
        filenames = [f"4_5/{i:06d}.parquet" for i in range(1, 1300)]
        finewebDataset = datasets.load_dataset(
            "opencsg/Fineweb-Edu-Chinese-V2.1",
            data_files=filenames,
            trust_remote_code=True,
            split="train"
        )
        finewebDataset = finewebDataset.map(
            functools.partial(alignedChunk, self.tokenizer, self.maxWindowSize, ["text"]),
            remove_columns=finewebDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return finewebDataset
    
    def addZhihuDataset(self):
        zhihuDataset = datasets.load_dataset(
            "wangrui6/Zhihu-KOL",
            split="train"
        )
        zhihuDataset = zhihuDataset.map(
            functools.partial(
                alignedChunk,
                self.tokenizer,
                self.maxWindowSize,
                ["INSTRUCTION", "RESPONSE"],
            ),
            remove_columns=zhihuDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return zhihuDataset
    
    def addCQIADataset(self):
        # add SFT data as well
        subsets =["chinese_traditional","douban","exam","logi_qa","zhihu","human_value"]
        ds = []
        for subset in subsets:
            d = datasets.load_dataset(
                "m-a-p/COIG-CQIA",
                subset,
                split="train"
            )
            ds.append(d)
        cqiaDataset =  datasets.concatenate_datasets(ds)
        cqiaDataset = cqiaDataset.map(
            functools.partial(alignedChunk, self.tokenizer, self.maxWindowSize, ["instruction", "output"]),
            remove_columns=cqiaDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return cqiaDataset
    
    def addCodeDataset(self):
        # 300m per file
        codeDataset = datasets.load_dataset(
            "bigcode/starcoderdata",
            data_files=[
                "python/train-00000-of-00059.parquet",
                "python/train-00001-of-00059.parquet",
                "python/train-00002-of-00059.parquet",
                "python/train-00003-of-00059.parquet",
                "python/train-00004-of-00059.parquet",
                "python/train-00005-of-00059.parquet",
                "python/train-00006-of-00059.parquet",
                "python/train-00007-of-00059.parquet",
                "python/train-00008-of-00059.parquet",
                "python/train-00009-of-00059.parquet",
                "python/train-00010-of-00059.parquet",
                "python/train-00011-of-00059.parquet",
                "python/train-00012-of-00059.parquet",
                "python/train-00013-of-00059.parquet",
                "python/train-00014-of-00059.parquet",
                "python/train-00015-of-00059.parquet",
            ],
            trust_remote_code=True,
            split="train"
        )
        codeDataset = codeDataset.map(
            functools.partial(alignedChunk, self.tokenizer, self.maxWindowSize, ["content"]),
            remove_columns=codeDataset.column_names,
            batched=True,
            num_proc=8,
            batch_size=30000,
        )
        return codeDataset

    def formatData(self, text):
        return {"text": text["text"]}

    def getTrainDataset(self):
        return self.trainDataset

    def getValDataset(self):
        return self.valDataset

    def printStats(self):
        totalChunks = 0
        for k in self.allDatasets:
            totalChunks += len(self.allDatasets[k])
        stats = {
            "event": "dataset",
            "trainChunks": totalChunks,
        }
        for k in self.allDatasets:
            stats[f"{k}Chunks"] = len(self.allDatasets[k])
            stats[f"{k}Tokens"] = len(self.allDatasets[k]) * self.maxWindowSize
            stats[f"{k}Ratio"] = len(self.allDatasets[k])/totalChunks*100
        util.log(stats)

if __name__ == "__main__":
    import scholar

    config = scholar.createModelConfig()
    config["maxWindowSize"] = 256
    tok = tokenizer.createTokenizer(config["tokenizer"])
    print(f"@@ Tokenizer: {tok}")
    ds = MixedDataLoader(config, tok, 1)
    ds.printStats()
    for i, batch in enumerate(ds.getTrainDataset()):
        if i < 5:
            print(f"@@{tok.decode(batch["inputs"])}")
        else:
            break
