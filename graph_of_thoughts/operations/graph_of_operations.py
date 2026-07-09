# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

from __future__ import annotations
from typing import List

from graph_of_thoughts.operations.operations import Operation


class GraphOfOperations:
    """
    表示 Graph of Operations，用于规定 thought operations 的执行计划。
    """

    def __init__(self) -> None:
        """
        初始化新的 Graph of Operations 实例，并设置空的 operations、roots 和 leaves。
        roots 是图中没有前驱的入口点。
        leaves 是图中没有后继的出口点。
        """
        self.operations: List[Operation] = []
        self.roots: List[Operation] = []
        self.leaves: List[Operation] = []

    def append_operation(self, operation: Operation) -> None:
        """
        将一个 operation 追加到图中的所有叶子节点之后，并更新关系。

        :param operation: 要追加的 operation。
        :type operation: Operation
        """
        self.operations.append(operation)

        if len(self.roots) == 0:
            self.roots = [operation]
        else:
            for leave in self.leaves:
                leave.add_successor(operation)

        self.leaves = [operation]

    def add_operation(self, operation: Operation) -> None:
        """
        在考虑前驱和后继的情况下向图中添加 operation。
        根据新增 operation 在图中的位置调整 roots 和 leaves。

        :param operation: 要添加的 operation。
        :type operation: Operation
        """
        self.operations.append(operation)
        if len(self.roots) == 0:
            self.roots = [operation]
            self.leaves = [operation]
            assert (
                len(operation.predecessors) == 0
            ), "First operation should have no predecessors"
        else:
            if len(operation.predecessors) == 0:
                self.roots.append(operation)
            for predecessor in operation.predecessors:
                if predecessor in self.leaves:
                    self.leaves.remove(predecessor)
            if len(operation.successors) == 0:
                self.leaves.append(operation)
