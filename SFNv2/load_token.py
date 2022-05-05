import os

import numpy as np
import re

current_path1 = os.path.abspath(__file__)
print(current_path1)

# re.search('^[0-9]')

token_arr = np.loadtxt(
    './multiscale_dataset/cgd/cwe119_cgd.txt',
    dtype=str,
    # comments=re.search('^[0-9]'),
    comments='-',
    delimiter='---------------------------------',
    encoding='UTF-8'
)

print(token_arr[6] == "0")

n = 0
token_list = []

str = ""
for t in token_arr:
    if t != "0" and t != "1":
        str += t
    else:
        token_list.append(str)
        str = ""


for i in range(10):
    print(token_list[i])




