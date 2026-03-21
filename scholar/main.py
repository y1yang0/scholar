# Copyright (c) 2026 yyang. All rights reserved.
import scholar
import util
import time
import sys
import json

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
    "徐凤年对徐渭熊说：",
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "train":
            g = scholar.Scholar(scholar.createModelConfig())
            resume = "" if len(sys.argv) <= 2 else sys.argv[2]
            g.train(resume)
        elif mode == "predict":
            texts = []
            if len(sys.argv) > 2:
                texts.append(sys.argv[2])
            else:
                texts = Sentences
            g = scholar.Predictor(scholar.createModelConfig(), "scholar_best.bin")
            for text in texts:
                response = g.predict(text)
                print(f"@@ {text}[{response}]")
        elif mode == "debug":
            sentence = "过儿"
            scholar.IsDebug = True
            print(f"@@ DebugInput: {sentence}")
            g = scholar.Predictor(scholar.createModelConfig(), "scholar_best.bin")
            output = g.predict(sentence, numNextToken=1)
            print(f"@@ DebugOutput: {output}")
    else:
        print("       python main.py [train|predict|debug]")
