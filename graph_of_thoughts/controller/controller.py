# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

import json
import logging
from typing import List
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations import GraphOfOperations, Thought
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser


class Controller:
    """
    用于管理 Graph of Operations 执行流程的 Controller 类，
    负责生成 Graph Reasoning State。
    该流程涉及语言模型、图操作、提示词构造和响应解析。
    """

    def __init__(
        self,
        lm: AbstractLanguageModel,
        graph: GraphOfOperations,
        prompter: Prompter,
        parser: Parser,
        problem_parameters: dict,
        skip_operation_ids: set = None,
        skip_operation_indices: set = None,
        skip_marker: str = "[SKIP]",
    ) -> None:
        """
        使用语言模型、operations graph、prompter、parser 和问题参数初始化 Controller 实例。

        :param lm: AbstractLanguageModel 的实例。
        :type lm: AbstractLanguageModel
        :param graph: 要执行的 Graph of Operations。
        :type graph: OperationsGraph
        :param prompter: Prompter 类的实例，用于生成提示词。
        :type prompter: Prompter
        :param parser: Parser 类的实例，用于解析响应。
        :type parser: Parser
        :param problem_parameters: 问题的初始参数或 state。
        :type problem_parameters: dict
        :param skip_operation_ids: 要替换为 skip 占位符的 operation id。
        :type skip_operation_ids: set
        :param skip_operation_indices: graph.operations 中要跳过的 operation 位置。
        :type skip_operation_indices: set
        :param skip_marker: 跳过节点使用的占位文本。
        :type skip_marker: str
        """
        self.logger = logging.getLogger(self.__class__.__module__)
        self.lm = lm
        self.graph = graph
        self.prompter = prompter
        self.parser = parser
        self.problem_parameters = problem_parameters
        self.skip_operation_ids = skip_operation_ids or set()
        self.skip_operation_indices = skip_operation_indices or set()
        self.skip_marker = skip_marker
        self.run_executed = False

    def run(self) -> None:
        """
        运行 controller，并根据 operation 是否就绪来执行 Graph of Operations 中的操作。
        执行前会确保程序处于有效状态。
        :raises AssertionError: 如果 Graph of Operation 没有 root。
        :raises AssertionError: 如果某个 operation 的后继不在 Graph of Operations 中。
        """
        self.logger.debug("Checking that the program is in a valid state")
        assert self.graph.roots is not None, "The operations graph has no root"
        self.logger.debug("The program is in a valid state")

        execution_queue = [
            operation
            for operation in self.graph.operations
            if operation.can_be_executed()
        ]

        while len(execution_queue) > 0:
            current_operation = execution_queue.pop(0)
            self.logger.info("Executing operation %s", current_operation.operation_type)
            current_operation_index = self.graph.operations.index(current_operation)
            if (
                current_operation.id in self.skip_operation_ids
                or current_operation_index in self.skip_operation_indices
            ):
                self.logger.info(
                    "Skipping operation %s at index %d",
                    current_operation.operation_type,
                    current_operation_index,
                )
                current_operation.skip(self.skip_marker, **self.problem_parameters)
            else:
                current_operation.execute(
                    self.lm, self.prompter, self.parser, **self.problem_parameters
                )
            self.logger.info("Operation %s executed", current_operation.operation_type)
            for operation in current_operation.successors:
                assert (
                    operation in self.graph.operations
                ), "The successor of an operation is not in the operations graph"
                if operation.can_be_executed():
                    execution_queue.append(operation)
        self.logger.info("All operations executed")
        self.run_executed = True

    def get_final_thoughts(self) -> List[List[Thought]]:
        """
        获取所有 operation 执行完成后的最终 thoughts。

        :return: 图中每个叶子 operation 的 thoughts 列表。
        :rtype: List[List[Thought]]
        :raises AssertionError: 如果尚未执行 `run` 方法。
        """
        assert self.run_executed, "The run method has not been executed"
        return [operation.get_thoughts() for operation in self.graph.leaves]

    def output_graph(self, path: str) -> None:
        """
        将 operations graph 的状态和结果序列化为 JSON 文件。

        :param path: 输出文件路径。
        :type path: str
        """
        output = []
        for operation in self.graph.operations:
            thoughts = operation.get_thoughts()
            # thought_metadata 与 thoughts 一一对应，保存每个 Thought 的观测信息：
            # 包括模型、token 用量、延迟、logprob/entropy、prompt、response 等。
            thought_metadata = [thought.metadata for thought in thoughts]
            operation_serialized = {
                # operation_id: 当前 operation 在本次图执行中的唯一编号。
                "operation_id": operation.id,
                # operation_index: operation 在 graph.operations 中的位置；跨多次建图更稳定。
                "operation_index": self.graph.operations.index(operation),
                # operation: operation 类型，例如 generate / aggregate / score。
                "operation": operation.operation_type.name,
                # predecessors/successors: 图拓扑关系，用于分析节点位置和结构重要性。
                "predecessors": [
                    predecessor.id for predecessor in operation.predecessors
                ],
                "successors": [successor.id for successor in operation.successors],
                # thoughts: 当前 operation 产生或保留的状态，由 parser 解析得到。
                "thoughts": [thought.state for thought in thoughts],
                # thought_metadata: 与 thoughts 对齐的观测数据，用于后续节点熵实验。
                "thought_metadata": thought_metadata,
            }
            if any([thought.scored for thought in thoughts]):
                operation_serialized["scored"] = [
                    thought.scored for thought in thoughts
                ]
                operation_serialized[("scor"
                                      "es")] = [thought.score for thought in thoughts]
            if any([thought.validated for thought in thoughts]):
                operation_serialized["validated"] = [
                    thought.validated for thought in thoughts
                ]
                operation_serialized["validity"] = [
                    thought.valid for thought in thoughts
                ]
            if any(
                [
                    thought.compared_to_ground_truth
                    for thought in thoughts
                ]
            ):
                operation_serialized["compared_to_ground_truth"] = [
                    thought.compared_to_ground_truth
                    for thought in thoughts
                ]
                operation_serialized["problem_solved"] = [
                    thought.solved for thought in thoughts
                ]
            output.append(operation_serialized)

        output.append(
            {
                "prompt_tokens": self.lm.prompt_tokens,
                "completion_tokens": self.lm.completion_tokens,
                "total_tokens": self.lm.prompt_tokens + self.lm.completion_tokens,
                "api_calls": getattr(self.lm, "api_calls", None),
                "total_latency_seconds": getattr(
                    self.lm, "total_latency_seconds", None
                ),
                "cost": self.lm.cost,
            }
        )

        with open(path, "w") as file:
            file.write(json.dumps(output, indent=2))
