# Copyright (c) 2026 yyang. All rights reserved.
import scholar
import util
import time
import sys
import json

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        util.log({"event": "start", "mode": mode, "timestamp": time.time()})
        g = scholar.Scholar(scholar.createModelConfig())
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
            scholar.IsDebug = True
            print(f"@@ DebugInput: {sentence}")
            [(_, output)] = g.predict([sentence], numNextToken=1)
            print(f"@@ DebugOutput: {output}")
    else:
        print("       python scholar.py [train|predict|tuning|debug]")
