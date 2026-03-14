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


def log(kvals):
    data = json.dumps(kvals, ensure_ascii=False)
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(data + "\n")


def listFiles(dirs):
    allFiles = []
    for d in dirs:
        for root, _, files in os.walk(d):
            for file in files:
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

    def start(self, tokenizer, inputTokens, out):
        self.inputTokens = inputTokens
        self.out = out
        self.tokenizer = tokenizer

    def trace(self, name, x, top=5):
        # for the given input tokens and intermediate tensor x, find the most
        # similar tokens
        print(f"@@ {name}:")
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
            print(f"   {inputToken} {similar}")
