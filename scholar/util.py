# Copyright (c) 2026 yyang. All rights reserved.
import json
import torch
import os


def getTorchDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cudaAvailable():
    return torch.cuda.is_available()


def log(kvals, file="train.log"):
    data = json.dumps(kvals, ensure_ascii=False)
    with open(file, "a", encoding="utf-8") as f:
        f.write(data + "\n")


def listFiles(dirs, includes=".txt", exclude=[]):
    allFiles = []
    for d in dirs:
        for root, _, files in os.walk(d):
            for file in files:
                if len(exclude) >0:
                    shouldExclude = False
                    for e in exclude:
                        if file.endswith(e):
                            shouldExclude = True
                            break
                    if shouldExclude:
                        continue
                if file.endswith(includes):
                    filePath = os.path.join(root, file)
                    allFiles.append(filePath)
    return allFiles


class Inspector:
    """
    Inspect model internal states
    """

    def __init__(self):
        self.inputTokens = []
        self.out = None
        self.started = False
        self.file = "inspect.log"

    def start(self, tokenizer, inputTokens, out):
        self.inputTokens = inputTokens
        self.out = out
        self.tokenizer = tokenizer
        if os.path.exists(self.file):
            os.remove(self.file)
        self.started = True

    def trace(self, name, x, top=15):
        if not self.started:
            return
        # x is in shape of (batchSize, inputLen, dimEmb)
        # for the given input tokens and intermediate tensor, find the most
        # similar tokens
        for t in range(x.shape[-2]):
            tokenEmbedding = x[0, t, :]
            logits = self.out(tokenEmbedding)
            topLogits, topIndices = torch.topk(logits, top)
            similar = []
            for k in range(topIndices.shape[0]):
                tokenId = topIndices[k].item()
                d = self.tokenizer.decode([tokenId])
                similar.append(f"{d}({topLogits[k].item():.1f})")

            inputToken = self.tokenizer.decode([self.inputTokens[t]])
            log(
                {"layer": name, "inputToken": inputToken, "similar": similar}, self.file
            )
