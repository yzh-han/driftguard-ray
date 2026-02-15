import time

start = time.perf_counter()
# 你的训练代码
time.sleep(2)  # 模拟训练过程
elapsed = time.perf_counter() - start
print(f"train time: {elapsed}s")

d = {"a": 1, "b": 2}