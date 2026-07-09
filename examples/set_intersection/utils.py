# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 本源代码改编自 Nils Blach 编写的 sorting 源代码。
#
# 主要作者：Robert Gerstenberger

from typing import Dict, List, Set


def string_to_list(string: str) -> List[int]:
    """将字符串中编码的列表转换为 Python 列表对象的辅助函数。"""

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return [int(num) for num in string[1:-1].split(",")]


def string_to_set(string: str) -> Set[int]:
    """将字符串中编码的列表转换为 Python 集合对象的辅助函数。"""

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return {int(num) for num in string[1:-1].split(",")}


def test_set_intersection(state: Dict) -> bool:
    """测试最终解是否与 ground truth 匹配的函数。"""

    # 将字符串转换为列表
    try:
        correct_list = string_to_list(state["result"])
        sorted_list = sorted(string_to_list(state["current"]))
        return sorted_list == correct_list
    except:
        return False


def num_errors(state: Dict) -> float:
    """本地统计错误数量并将其作为分数的函数。"""

    try:
        set1 = string_to_set(state["set1"])
        set2 = string_to_set(state["set2"])
        if "subset" in state and state["subset"] != "" and state["subset"] is not None:
            set2 = string_to_set(state["subset"])
        common = sorted(list(set1 & set2))
        llm_solution = sorted(string_to_list(state["current"]))
        num_errors = 0
        common_idx = 0
        llm_idx = 0
        while common_idx < len(common) and llm_idx < len(llm_solution):
            if common[common_idx] == llm_solution[llm_idx]:
                common_idx += 1
                llm_idx += 1
            elif common[common_idx] < llm_solution[llm_idx]:
                common_idx += 1
                num_errors += 1
            elif common[common_idx] > llm_solution[llm_idx]:
                llm_idx += 1
                num_errors += 1
        num_errors += len(common) - common_idx + len(llm_solution) - llm_idx
        return num_errors
    except:
        return 1000
