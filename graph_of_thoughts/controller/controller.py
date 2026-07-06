# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import json
import logging
from typing import List
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations import GraphOfOperations, Thought
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser


class Controller:
    """
    Controller class to manage the execution flow of the Graph of Operations,
    generating the Graph Reasoning State.
    This involves language models, graph operations, prompting, and parsing.
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
        Initialize the Controller instance with the language model,
        operations graph, prompter, parser, and problem parameters.

        :param lm: An instance of the AbstractLanguageModel.
        :type lm: AbstractLanguageModel
        :param graph: The Graph of Operations to be executed.
        :type graph: OperationsGraph
        :param prompter: An instance of the Prompter class, used to generate prompts.
        :type prompter: Prompter
        :param parser: An instance of the Parser class, used to parse responses.
        :type parser: Parser
        :param problem_parameters: Initial parameters/state of the problem.
        :type problem_parameters: dict
        :param skip_operation_ids: Operation ids to replace with a skip placeholder.
        :type skip_operation_ids: set
        :param skip_operation_indices: Operation positions in graph.operations to skip.
        :type skip_operation_indices: set
        :param skip_marker: Placeholder text used for skipped nodes.
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
        Run the controller and execute the operations from the Graph of
        Operations based on their readiness.
        Ensures the program is in a valid state before execution.
        :raises AssertionError: If the Graph of Operation has no roots.
        :raises AssertionError: If the successor of an operation is not in the Graph of Operations.
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
        Retrieve the final thoughts after all operations have been executed.

        :return: List of thoughts for each operation in the graph's leaves.
        :rtype: List[List[Thought]]
        :raises AssertionError: If the `run` method hasn't been executed yet.
        """
        assert self.run_executed, "The run method has not been executed"
        return [operation.get_thoughts() for operation in self.graph.leaves]

    def output_graph(self, path: str) -> None:
        """
        Serialize the state and results of the operations graph to a JSON file.

        :param path: The path to the output file.
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
                operation_serialized["scores"] = [thought.score for thought in thoughts]
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
                "cost": self.lm.cost,
            }
        )

        with open(path, "w") as file:
            file.write(json.dumps(output, indent=2))
