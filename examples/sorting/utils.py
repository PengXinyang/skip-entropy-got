# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

from typing import Dict, List


def string_to_list(string: str) -> List[int]:
    """将字符串中编码的列表转换为 Python 列表对象的辅助函数。"""

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return [int(num) for num in string[1:-1].split(",")]


def test_sorting(state: Dict) -> bool:
    """测试最终解是否与 ground truth 匹配的函数。"""

    try:
        correct_list = sorted(string_to_list(state["original"]))
        sorted_list = string_to_list(state["current"])
        return sorted_list == correct_list
    except:
        return False


def num_errors(state: Dict) -> float:
    """本地统计错误数量并将其作为分数的函数。"""

    try:
        unsorted_list = state["original"]
        if (
            "unsorted_sublist" in state
            and state["unsorted_sublist"] != ""
            and state["unsorted_sublist"] is not None
            and len(state["unsorted_sublist"]) < len(unsorted_list) - 5
        ):
            unsorted_list = state["unsorted_sublist"]
        correct_list = sorted(string_to_list(unsorted_list))
        current_list = string_to_list(state["current"])
        num_errors = 0
        for i in range(10):
            num_errors += abs(
                sum([1 for num in current_list if num == i])
                - sum([1 for num in correct_list if num == i])
            )
        num_errors += sum(
            [1 for num1, num2 in zip(current_list, current_list[1:]) if num1 > num2]
        )
        return num_errors
    except:
        return 300
