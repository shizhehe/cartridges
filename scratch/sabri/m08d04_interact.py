import time

print("Starting...")
for i in range(100_000):
    print(f"Iteration {i} at {time.time()}")
    time.sleep(60)
    