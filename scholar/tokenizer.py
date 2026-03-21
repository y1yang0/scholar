# Copyright (c) 2026 yyang. All rights reserved.
import sys
import os
import util
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import transformers
import scholar
import pathlib

TrainedTokenizerFile = "tokenizer.json"
MaxVocabSize = 25000
ScriptDir = pathlib.Path(__file__).parent


class InternMLTokenizer:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("internlm/internlm-7b", add_bos_token=False, add_eos_token=False, trust_remote_code=True)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocabSize(self):
        return len(self.tokenizer)

    def endOfText(self):
        # eot is guaranteed to be ONE special token
        eosTokenId = self.tokenizer.eos_token_id
        return (eosTokenId, self.tokenizer.decode([eosTokenId]))

# Self-trained tokenizer for Jinyong-specific dataset, from huggingface/tokenizer
class BBPETokenizer:
    def __init__(self):
        path = util.getConfigPath(TrainedTokenizerFile)
        self.tokenizer = Tokenizer.from_file(path)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def vocabSize(self):
        return self.tokenizer.get_vocab_size()

    def endOfText(self):
        # eot is guaranteed to be ONE special token
        return (self.encode(EOT)[0], EOT)


def createSpecialTokens(files):
    specialTokens = []
    for file in files:
        name = os.path.basename(file).split(".")[0]
        specialTokens.append(f"<|{name}|>")
    return specialTokens


def trainTokenizer():
    files = util.listFiles(scholar.createModelConfig()["dataset"], ".txt", [])
    print(f"@@ Training tokenizer with {len(files)} files")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    vocabSize = MaxVocabSize + (64 - MaxVocabSize % 64)
    specialTokens = ["<|pad|>", "<|cls|>", "<|sep|>", "<|mask|>", EOT]
    specialTokens = specialTokens + createSpecialTokens(files)
    print(f"@@ Special tokens: {specialTokens}")
    trainer = trainers.BpeTrainer(
        vocab_size=vocabSize,
        special_tokens=specialTokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files, trainer)
    path = util.getConfigPath(TrainedTokenizerFile)
    tokenizer.save(path)
    print(f"@@ Save tokenizer to {path}")


def testTokenizer():
    path = util.getConfigPath(TrainedTokenizerFile)
    tokenizer = Tokenizer.from_file(path)
    sentences = [
        "杨过",
        "杨过和小龙女在",
        "神雕大侠",
        "韦小宝和双儿",
        "围攻光明顶",
        "郭靖和黄蓉",
        "张无忌",
        "令狐冲说",
        "华山论剑",
        "桃花岛上",
        "少林寺",
        "降龙十八掌",
        "他使出一剑",
        "女子冷哼一声",
        "你懂剑法吗？",
        "一剑刺中了",
    ]
    print(f"@@ Testing tokenizer with {sentences} sentences")
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        print(f"@@ Raw:      {sentence}")
        print(f"@@ Encoded: '{encoded.tokens}'")
        decodeTokens = [tokenizer.decode([i]) for i in encoded.ids]
        print(f"@@ Decoded: '{decodeTokens}'")

def createTokenizer(name):
    registry = {
        "InternMLTokenizer": InternMLTokenizer,
        "BBPETokenizer": BBPETokenizer,
    }
    if name not in registry:
        raise RuntimeError(f"Unknown tokenizer {name}, available: {list(registry.keys())}")
    return registry[name]()


def testInternMLTokenizer():
    tokenizer = InternMLTokenizer()
    sentences = [
        "杨过",
        "杨过和小龙女在",
        "神雕大侠",
    ]
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        print(f"@@ Raw:      {sentence}")
        print(f"@@ Encoded: '{encoded}'")
        decoded = tokenizer.decode(encoded)
        print(f"@@ Decoded: '{decoded}'")
    print(f"@@ Vocab size: {tokenizer.vocabSize()}")
    print(f"@@ End of text: {tokenizer.endOfText()}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainTokenizer()
    else:
        testTokenizer()
        testInternMLTokenizer()
