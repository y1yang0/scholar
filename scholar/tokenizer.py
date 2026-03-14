# Copyright (c) 2026 yyang. All rights reserved.
import sys
import util
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import scholar

eot = "<|endoftext|>"


# Self-trained tokenizer for Jinyong-specific dataset, from huggingface/tokenizer
class BBPETokenizer:
    def __init__(self):
        path = "scholar/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(path)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def vocabSize(self):
        return self.tokenizer.get_vocab_size()

    def endOfText(self):
        # eot is guaranteed to be ONE special token
        return (self.encode(eot)[0], eot)


def trainTokenizer():
    files = util.listFiles(scholar.createModelConfig()["dataset"])
    print(f"@@ Training tokenizer with {files} files")
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=20000,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", eot],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files, trainer)
    path = "scholar/tokenizer.json"
    tokenizer.save(path)
    print(f"@@ Save tokenizer to {path}")


def testTokenizer():
    path = "scholar/tokenizer.json"
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainTokenizer()
    else:
        testTokenizer()
