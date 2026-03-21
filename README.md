<div  align="center">
<img src="misc/logo.png" width="20%">

**Scholar Etude** - Modernized minimal GPT implementation from scratch.
</div>

----

**Scholar Dashboard**

Just run `python scholar/dashboard.py` and open `http://localhost:5002` in your browser.

![](misc/dashboard1.png)
![](misc/dashboard2.png)
![](misc/dashboard3.png)

---

**Scholar Commands**

**Install dependencies**
```
$ pip install -r requirements.txt
```
GPU device is also required.

**Pre-training the model:**
```
#  torchrun --nproc_per_node=1 scholar/main.py train
```
Use `CUDA_VISIBLE_DEVICES=0` to specify the certain GPU device to use.
Alternatively, you can use dashboard to start training. It's the recommended way.

**Generating next few words:**
```
$ python scholar/main.py predict "杨过的姑姑"
```

**Inspecting model internals:**
```
$ python scholar/main.py debug
```
