# Copyright (c) 2026 yyang. All rights reserved.
import torch
import os
import random
import json
import util


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

    def __init__(self, config, tokenizer, files):
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


class LargeDataLoader:
    """
    Load data streamly, used for large dataset, e.g. Wikipedia
    """

    def __init__(self, config, tokenizer, files):
        random.shuffle(files)
        trainDataRatio = config["trainDataRatio"]
        trainFiles = files[: int(len(files) * trainDataRatio)]
        valFiles = files[int(len(files) * trainDataRatio) :]
        self.trainLoader = StreamDataLoader(config, tokenizer, trainFiles)
        self.valLoader = StreamDataLoader(config, tokenizer, valFiles)

    def nextValBatch(self):
        return self.valLoader.nextBatch()

    def nextTrainBatch(self):
        return self.trainLoader.nextBatch()


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
                        inputBatch, targetBatch = packToBatch(chunkBuf)
                        chunkBuf = []
                        yield inputBatch, targetBatch
        # sft data is precious, don't let it go to waste
        if len(chunkBuf) > 0:
            inputBatch, targetBatch = packToBatch(chunkBuf)
            yield inputBatch, targetBatch


class MixedDataLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # prepare all kinds of datasets
        self.novelDataset = datasets.load_dataset(
            "y1yang0/scholar-novels-curated", split="train"
        )
        self.wikiDataset = datasets.load_dataset(
            "shaowenchen/wiki_zh", split="train[:5%]"
        )
        # make sure all datasets have the same format, this is required by the
        # interleave_datasets function
        formatData = lambda text: {"text": text["text"]}
        self.novelDataset = self.novelDataset.map(
            formatData, remove_columns=self.novelDataset.column_names
        )
        self.wikiDataset = self.wikiDataset.map(
            formatData, remove_columns=self.wikiDataset.column_names
        )
        # interleave the wiki and novel dataset with A% wiki and B% novel and etc
        self.mixedDataset = datasets.interleave_datasets(
            [self.wikiDataset, self.novelDataset],
            probabilities=[0.0, 1.0],
            seed=0xCAFEBABE,
        )
        # apply tokenization to the mixed dataset
        splitDataset = self.mixedDataset.train_test_split(
            test_size=0.1, seed=0xCAFEBABE
        )
        self.trainDataset = splitDataset["train"]
        self.valDataset = splitDataset["test"]

    def printDataset(self):
        totalRow = len(self.novelDataset) + len(self.wikiDataset)
        print("@@ Novel Dataset:")
        print(self.novelDataset)
        print(self.novelDataset.features)
        for i in range(2):
            print(f"@@    Text Sample {i}: {self.novelDataset[i]['text'][:100]}")
        print("@@ Wiki Dataset:")
        print(self.wikiDataset)
        # for i in range(2):
        # print(f"@@    TextSample {i}: {self.wikiDataset[i]['text'][:100]}")
        print(f"@@ Total rows: {totalRow}")
        print(f"@@ Novel ratio: {len(self.novelDataset)/totalRow*100:.2f}%")
        print(f"@@ Wiki ratio: {len(self.wikiDataset)/totalRow*100:.2f}%")
        print(f"@@ Train dataset: {len(self.trainDataset)}")
        print(f"@@ Val dataset: {len(self.valDataset)}")


# def testHFDataset():
#     dataset = datasets.load_dataset("y1yang0/scholar-novels-curated", split=["train","validation"])
#     print(dataset)
#     print(dataset["train"][0])
#     print(dataset["train"][1])
#     print(dataset["train"][2])
#     print(dataset["train"][3])
#     print(dataset["train"][4])
#     print(dataset["train"][5])
#     print(dataset["train"][6])
#     print(dataset["train"][7])
#     print(dataset["train"][8])
#     print(dataset["train"][9])

if __name__ == "__main__":
    import scholar

    tokenizer = tokenizer.BBPETokenizer()
    ds = MixedDataLoader(scholar.createModelConfig(), tokenizer)
    ds.printDataset()
    idx = 0
    for textIds in ds.mixedDataset:
        idx += 1
