"""Minimal multiprocessing test on Windows."""
import concurrent.futures as cf
import numpy as np

ARR = np.array([1.0, 2.0, 3.0])

def worker(x):
    return float(x * np.sum(ARR))

if __name__ == "__main__":
    tasks = list(range(8))
    with cf.ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, tasks))
    print(f"Results: {results}")
    print("PASS")
