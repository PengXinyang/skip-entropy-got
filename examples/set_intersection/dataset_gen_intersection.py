# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Robert Gerstenberger

import csv
import numpy as np


def scramble(array: np.ndarray, rng: np.random.Generator) -> None:
    """随机改变数组中元素顺序的辅助函数。"""

    size = array.shape[0]

    index_array = rng.integers(0, size, size)

    for i in range(size):
        temp = array[i]
        array[i] = array[index_array[i]]
        array[index_array[i]] = temp


if __name__ == "__main__":
    """
    Input(u)  : ?????
    Input(v)  : ???????????0..v???? v??
    Input(w)  : ??????????
    Input(x)  : ?????????
    Input(y)  : ?? CSV ????
    Output(z) : ? CSV ?????????????????
                ?????? ID????? 1????? 2 ??????
    """

    set_size = 32  # ???????
    int_value_ubound = 64  # 检查无效国家和形容词???
    seed = 42  # 将字符串转换为列表
    num_sample = 100  # ????
    filename = "set_intersection_032.csv"  # ?????

    assert 2 * set_size <= int_value_ubound

    rng = np.random.default_rng(seed)

    intersection_sizes = rng.integers(set_size // 4, 3 * set_size // 4, num_sample)

    np.set_printoptions(
        linewidth=np.inf
    )  # 检查无效国家和形容词???

    with open(filename, "w") as f:
        fieldnames = ["ID", "SET1", "SET2", "INTERSECTION"]
        writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_sample):
            intersection_size = intersection_sizes[i]

            full_set = np.arange(0, int_value_ubound, dtype=np.int16)

            scramble(full_set, rng)

            intersection = full_set[:intersection_size].copy()

            sorted_intersection = np.sort(intersection)

            set1 = full_set[:set_size].copy()
            set2 = np.concatenate(
                [intersection, full_set[set_size : 2 * set_size - intersection_size]]
            )

            scramble(set1, rng)
            scramble(set2, rng)

            writer.writerow(
                {
                    "ID": i,
                    "SET1": set1.tolist(),
                    "SET2": set2.tolist(),
                    "INTERSECTION": sorted_intersection.tolist(),
                }
            )
