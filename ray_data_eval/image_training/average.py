import numpy as np

f = open("temp.txt", "r")
arr = np.array([])
for line in f:
    name, res = line.split(',')
    res = float(res[:-1])
    arr = np.append(arr, res)

print(np.mean(arr))
