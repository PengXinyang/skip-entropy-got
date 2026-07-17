# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

from __future__ import annotations
import logging
from enum import Enum
from typing import List, Iterator, Dict, Callable, Union, Any, Tuple
from abc import ABC, abstractmethod
import itertools

from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser


def _operation_response_metadata(
    lm: AbstractLanguageModel,
    responses: List[str],
    prompt: str,
    operation: "Operation",
    prompt_role: str,
) -> List[Dict[str, Any]]:
    """
    消费最近响应文本对应的模型元数据，并补充 operation 上下文。
    返回的列表会尽可能与 responses 对齐。
    """
    metadata = lm.consume_last_response_metadata(len(responses))
    for index, item in enumerate(metadata):
        item.update(
            {
                # operation_id: 当前 GoT operation 的唯一编号，可用于定位图中节点。
                "operation_id": operation.id,
                # operation_type: 当前节点类型，例如 generate / aggregate / score。
                "operation_type": operation.operation_type.name,
                # prompt_role: 这次 LLM 调用在流程里的用途，例如 generate/aggregate。
                "prompt_role": prompt_role,
                # prompt: 发送给模型的完整提示词，便于复现实验和排查失败案例。
                "prompt": prompt,
                # response_text: 模型原始文本回复；parser 会从这里解析出 thought.state。
                "response_text": responses[index] if index < len(responses) else None,
                # predecessor_operation_ids: 当前 operation 依赖的上游 operation id。
                "predecessor_operation_ids": [
                    predecessor.id for predecessor in operation.predecessors
                ],
            }
        )
    return metadata


def _metadata_for_created_thought(
    response_metadata: List[Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    """
    为解析出的 thought 选择元数据。部分 parser 会把一个 response 拆成多个 thought；
    这种情况下，每个 thought 都继承这条 response 的元数据。
    """
    # 一个 response 可能被 parser 拆成多个 Thought，例如一次 split prompt 返回多个子列表。
    # 如果只有一条 response metadata，就让拆出来的多个 Thought 共享这份观测信息。
    if not response_metadata:
        return {}
    if len(response_metadata) == 1:
        return dict(response_metadata[0])
    if index < len(response_metadata):
        return dict(response_metadata[index])
    return dict(response_metadata[-1])


def _create_skipped_thought(
    base_state: Dict,
    operation: "Operation",
    thought_index: int,
    skip_marker: str,
    prompt_role: str,
) -> Thought:
    """
    为 thought 级 skip 创建占位 Thought。

    图结构和 thought 位置保持不变，但该 thought 对应的 LLM 调用不会发生。
    """
    skipped_state = dict(base_state or {})
    skipped_state["current"] = skip_marker
    skipped_state["skipped"] = True
    skipped_state["skip_source_operation_id"] = operation.id
    skipped_state["skip_source_thought_index"] = thought_index
    return Thought(
        skipped_state,
        metadata={
            "skipped": True,
            "skip_marker": skip_marker,
            "skip_level": "thought",
            "operation_id": operation.id,
            "operation_type": operation.operation_type.name,
            "thought_index": thought_index,
            "prompt_role": prompt_role,
            "predecessor_operation_ids": [
                predecessor.id for predecessor in operation.predecessors
            ],
        },
    )


def _numeric_metadata_value(metadata: Dict[str, Any], field: str) -> Union[float, None]:
    """
    从 metadata 中安全读取数值字段。
    """
    value = metadata.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _force_single_response_calls(lm: AbstractLanguageModel) -> bool:
    """
    判断是否把一次 num_responses=N 的请求拆成 N 次 num_responses=1。

    该开关用于新的实验口径：统一“一个候选 thought/response 一次模型调用”，
    避免 GPT 的 n=N 调用和 DeepSeek/Gemini 的多次 n=1 调用在调用次数上不可比。
    """
    config = getattr(lm, "config", {}) or {}
    return bool(config.get("force_single_response_calls", False))


def _query_lm_with_metadata(
    lm: AbstractLanguageModel,
    prompt: str,
    num_responses: int,
    operation: "Operation",
    prompt_role: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    查询语言模型并返回与 responses 对齐的 metadata。

    默认保持原行为：一次请求传入 num_responses。
    当 force_single_response_calls=true 时，将其拆成多次 num_responses=1，
    从而让调用次数与候选 thought/response 数量一致。
    """
    if num_responses <= 1 or not _force_single_response_calls(lm):
        responses = lm.get_response_texts(lm.query(prompt, num_responses=num_responses))
        return responses, _operation_response_metadata(
            lm, responses, prompt, operation, prompt_role
        )

    all_responses: List[str] = []
    all_metadata: List[Dict[str, Any]] = []
    for response_index in range(num_responses):
        responses = lm.get_response_texts(lm.query(prompt, num_responses=1))
        response_metadata = _operation_response_metadata(
            lm, responses, prompt, operation, prompt_role
        )
        for item in response_metadata:
            # response_call_index: 拆分调用后的第几次请求，便于统计和排查。
            item["response_call_index"] = response_index
            # requested_num_responses: 原本希望从该 prompt 生成多少个候选。
            item["requested_num_responses"] = num_responses
            # force_single_response_calls: 标记本次 metadata 来自拆分调用模式。
            item["force_single_response_calls"] = True
        all_responses.extend(responses)
        all_metadata.extend(response_metadata)
    return all_responses, all_metadata


class OperationType(Enum):
    """
    表示不同 operation 类型的枚举，可作为唯一标识使用。
    """

    score: int = 0
    validate_and_improve: int = 1
    generate: int = 2
    improve: int = 3
    aggregate: int = 4
    keep_best_n: int = 5
    keep_valid: int = 6
    ground_truth_evaluator: int = 7
    selector: int = 8


class Operation(ABC):
    """
    定义所有 operation 接口的抽象基类。
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        初始化新的 Operation 实例，并设置唯一 id、空的前驱和后继列表。
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False
        # skip_thought_indices: 当前 operation 内需要跳过的 thought 序号。
        # Controller 会在执行前写入，Generate/Improve/Aggregate 在各自调用点消费。
        self.skip_thought_indices: set = set()
        # skip_refine_indices: ValidateAndImprove 内部 refine 事件的跳过集合。
        # 元素为 (input_thought_index, try_index)，表示第几个输入 thought 的第几次 refine。
        self.skip_refine_indices: set = set()
        self.skip_marker: str = "[SKIP]"

    def can_be_executed(self) -> bool:
        """
        根据前驱 operation 判断当前 operation 是否可以执行。

        :return: 如果所有前驱都已执行则为 True，否则为 False。
        :rtype: bool
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        """
        遍历所有前驱，并汇总它们的 thoughts。

        :return: 来自所有前驱的 thoughts 列表。
        :rtype: List[Thought]
        """
        previous_thoughts: List[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]

        return previous_thoughts

    def add_predecessor(self, operation: Operation) -> None:
        """
        添加一个前驱 operation，并更新两者之间的关系。

        :param operation: 要设置为前驱的 operation。
        :type operation: Operation
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: Operation) -> None:
        """
        添加一个后继 operation，并更新两者之间的关系。

        :param operation: 要设置为后继的 operation。
        :type operation: Operation
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行当前 operation，并确保所有前驱都已经执行。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果并非所有前驱都已执行。
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info(
            "Executing operation %d of type %s", self.id, self.operation_type
        )
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug("Operation %d executed", self.id)
        self.executed = True

    @abstractmethod
    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        实际执行 operation 的抽象方法。
        该方法应由派生类实现。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        """
        获取与当前 operation 关联的 thoughts 的抽象方法。
        该方法应由派生类实现。

        :return: 关联的 thoughts 列表。
        :rtype: List[Thought]
        """
        pass


class Score(Operation):
    """
    用于给 thoughts 打分的 operation。
    """

    operation_type: OperationType = OperationType.score

    def __init__(
        self,
        num_samples: int = 1,
        combined_scoring: bool = False,
        scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = None,
    ) -> None:
        """
        初始化新的 Score operation。

        :param num_samples: 打分时使用的样本数量。默认为 1。
        :type num_samples: int
        :param combined_scoring: 是否将所有 thoughts 一起打分，而不是逐个打分。默认为 False。
        :type combined_scoring: bool
        :param scoring_function: 不使用 LM 时用于给 thoughts 打分的函数。默认为 None。
        :type scoring_function: 接收 thought state 列表或单个 thought state，
                                并返回分数列表或单个分数。
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.thoughts: List[Thought] = []
        self.scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = scoring_function

    def get_thoughts(self) -> List[Thought]:
        """
        返回与当前 operation 关联的 thoughts。

        :return: 已打分的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过给前驱中的 thoughts 打分来执行 scoring operation。
        如果启用 combined scoring，则将所有 thoughts 一起打分，否则逐个打分。
        如果提供了 scoring function，则使用该函数；否则向 LM 发送提示。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "Score operation needs at least one predecessor"

        if self.combined_scoring:
            previous_thoughts_states = [thought.state for thought in previous_thoughts]
            if self.scoring_function is not None:
                self.logger.debug(
                    "Using scoring function %s to score states", self.scoring_function
                )
                scores = self.scoring_function(previous_thoughts_states)
            else:
                prompt = prompter.score_prompt(previous_thoughts_states)
                self.logger.debug("Prompt for LM: %s", prompt)

                responses, score_metadata = _query_lm_with_metadata(
                    lm, prompt, self.num_samples, self, "score"
                )
                self.logger.debug("Responses from LM: %s", responses)
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
            for thought, score in zip(previous_thoughts, scores):
                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                if self.scoring_function is None:
                    new_thought.metadata["score_observation"] = _metadata_for_created_thought(
                        score_metadata, len(self.thoughts)
                    )
                self.thoughts.append(new_thought)
        else:
            for thought in previous_thoughts:
                new_thought = Thought.from_thought(thought)
                if self.scoring_function is not None:
                    self.logger.debug(
                        "Using scoring function %s to score state",
                        self.scoring_function,
                    )
                    score = self.scoring_function(thought.state)
                else:
                    prompt = prompter.score_prompt([thought.state])
                    self.logger.debug("Prompt for LM: %s", prompt)

                    responses, score_metadata = _query_lm_with_metadata(
                        lm, prompt, self.num_samples, self, "score"
                    )
                    self.logger.debug("Responses from LM: %s", responses)
                    score = parser.parse_score_answer([thought.state], responses)[0]

                new_thought.score = score
                if self.scoring_function is None:
                    new_thought.metadata["score_observation"] = _metadata_for_created_thought(
                        score_metadata, 0
                    )
                self.thoughts.append(new_thought)

        self.logger.info(
            "Score operation %d scored %d thoughts",
            self.id,
            len(self.thoughts),
        )


class ValidateAndImprove(Operation):
    """
    用于验证并改进 thoughts 的 operation。
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(
        self,
        num_samples: int = 1,
        improve: bool = True,
        num_tries: int = 3,
        validate_function: Callable[[Dict], bool] = None,
    ) -> None:
        """
        初始化新的 ValidateAndImprove operation。

        :param num_samples: 验证时使用的样本数量。默认为 1。
        :type num_samples: int
        :param improve: 当 thought 无效时是否进行改进。默认为 True。
        :type improve: bool
        :param num_tries: 放弃前尝试改进 thought 的次数。默认为 3。
        :type num_tries: int
        :param validate_function: 不使用 LM 时用于验证 thoughts 的函数。默认为 None。
        :type validate_function: 接收 thought state 并返回布尔值。
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.improve: bool = improve
        self.num_tries: int = num_tries
        self.validate_function: Callable[[Dict], bool] = validate_function
        self.thoughts: List[List[Thought]] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回经过验证和改进后的最终 thoughts 列表。

        :return: 最终验证并改进后的 thoughts 列表。
        :rtype: List[Thought]
        """
        return [thought_list[-1] for thought_list in self.thoughts]

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过验证并改进前驱中的 thoughts 来执行 ValidateAndImprove operation。
        如果提供了 validation function，则使用该函数；否则向 LM 发送提示。
        如果启用了改进，并且 thought 无效，则向 LM 发送提示以改进该 thought。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "ValidateAndImprove operation needs at least one predecessor"

        for input_thought_index, thought in enumerate(previous_thoughts):
            thought_list = []
            current_thought = Thought.from_thought(thought)
            current_try = 0
            while True:
                if self.validate_function is not None:
                    self.logger.debug(
                        "Using validate function %s to score states",
                        self.validate_function,
                    )
                    valid = self.validate_function(current_thought.state)
                else:
                    prompt = prompter.validation_prompt(**current_thought.state)
                    self.logger.debug("Prompt for LM: %s", prompt)
                    responses, validation_metadata = _query_lm_with_metadata(
                        lm, prompt, self.num_samples, self, "validation"
                    )
                    self.logger.debug("Responses from LM: %s", responses)

                    valid = parser.parse_validation_answer(
                        current_thought.state, responses
                    )
                    current_thought.metadata["validation_observation"] = (
                        _metadata_for_created_thought(validation_metadata, 0)
                    )
                current_thought.valid = valid
                thought_list.append(current_thought)
                if (
                    not self.improve
                    or current_thought.valid
                    or current_try >= self.num_tries
                ):
                    break

                refine_key = (input_thought_index, current_try)
                if refine_key in self.skip_refine_indices:
                    skipped_state = dict(current_thought.state or {})
                    skipped_state["skipped"] = True
                    skipped_state["skip_role"] = "validate_and_improve_refine"
                    skipped_state["skip_source_operation_id"] = self.id
                    skipped_state["skip_source_input_thought_index"] = (
                        input_thought_index
                    )
                    skipped_state["skip_source_try_index"] = current_try
                    skipped_thought = Thought(
                        skipped_state,
                        metadata=dict(current_thought.metadata),
                    )
                    skipped_thought.metadata.update(
                        {
                            "skipped": True,
                            "skip_marker": self.skip_marker,
                            "skip_level": "validate_and_improve_refine",
                            "operation_id": self.id,
                            "operation_type": self.operation_type.name,
                            "input_thought_index": input_thought_index,
                            "try_index": current_try,
                            "prompt_role": "validate_and_improve_refine",
                            "skip_validate_after_refine": True,
                            "predecessor_operation_ids": [
                                predecessor.id
                                for predecessor in self.predecessors
                            ],
                        }
                    )
                    skipped_thought.validated = current_thought.validated
                    skipped_thought._valid = current_thought.valid
                    thought_list.append(skipped_thought)
                    break

                input_entropy = _numeric_metadata_value(
                    current_thought.metadata, "avg_entropy_bits"
                )
                input_normalized_entropy = _numeric_metadata_value(
                    current_thought.metadata, "normalized_avg_entropy_bits"
                )
                previous_refine_events = list(
                    current_thought.metadata.get(
                        "validate_and_improve_refine_events", []
                    )
                )
                improve_prompt = prompter.improve_prompt(**current_thought.state)
                self.logger.debug("Prompt for LM: %s", improve_prompt)
                responses, improve_metadata = _query_lm_with_metadata(
                    lm, improve_prompt, 1, self, "improve"
                )
                self.logger.debug("Responses from LM: %s", responses)
                state_update = parser.parse_improve_answer(
                    current_thought.state, responses
                )
                current_thought = Thought(
                    {**current_thought.state, **state_update},
                    metadata=_metadata_for_created_thought(improve_metadata, 0),
                )
                current_thought.metadata.update(
                    {
                        "input_thought_index": input_thought_index,
                        "try_index": current_try,
                        "prompt_role": "validate_and_improve_refine",
                    }
                )
                refined_entropy = _numeric_metadata_value(
                    current_thought.metadata, "avg_entropy_bits"
                )
                if input_entropy is not None and refined_entropy is not None:
                    delta_entropy = refined_entropy - input_entropy
                    current_thought.metadata["input_entropy_bits"] = input_entropy
                    current_thought.metadata["refined_entropy_bits"] = refined_entropy
                    current_thought.metadata["delta_entropy_bits"] = delta_entropy
                    current_thought.metadata["abs_delta_entropy_bits"] = abs(
                        delta_entropy
                    )
                refined_normalized_entropy = _numeric_metadata_value(
                    current_thought.metadata, "normalized_avg_entropy_bits"
                )
                if (
                    input_normalized_entropy is not None
                    and refined_normalized_entropy is not None
                ):
                    normalized_delta = (
                        refined_normalized_entropy - input_normalized_entropy
                    )
                    current_thought.metadata[
                        "input_normalized_avg_entropy_bits"
                    ] = input_normalized_entropy
                    current_thought.metadata[
                        "refined_normalized_avg_entropy_bits"
                    ] = refined_normalized_entropy
                    current_thought.metadata[
                        "delta_normalized_avg_entropy_bits"
                    ] = normalized_delta
                    current_thought.metadata[
                        "abs_delta_normalized_avg_entropy_bits"
                    ] = abs(normalized_delta)
                refine_event = {
                    "operation_id": self.id,
                    "operation_type": self.operation_type.name,
                    "input_thought_index": input_thought_index,
                    "try_index": current_try,
                    "event_type": "refine",
                    "entropy_field": "avg_entropy_bits",
                    "input_entropy_bits": input_entropy,
                    "refined_entropy_bits": refined_entropy,
                    "delta_entropy_bits": (
                        refined_entropy - input_entropy
                        if input_entropy is not None and refined_entropy is not None
                        else None
                    ),
                    "abs_delta_entropy_bits": (
                        abs(refined_entropy - input_entropy)
                        if input_entropy is not None and refined_entropy is not None
                        else None
                    ),
                    "input_normalized_avg_entropy_bits": input_normalized_entropy,
                    "refined_normalized_avg_entropy_bits": refined_normalized_entropy,
                    "delta_normalized_avg_entropy_bits": (
                        refined_normalized_entropy - input_normalized_entropy
                        if input_normalized_entropy is not None
                        and refined_normalized_entropy is not None
                        else None
                    ),
                    "abs_delta_normalized_avg_entropy_bits": (
                        abs(refined_normalized_entropy - input_normalized_entropy)
                        if input_normalized_entropy is not None
                        and refined_normalized_entropy is not None
                        else None
                    ),
                }
                refine_events = previous_refine_events
                refine_events.append(refine_event)
                current_thought.metadata[
                    "validate_and_improve_refine_events"
                ] = refine_events
                current_try += 1
            self.thoughts.append(thought_list)

        self.logger.info(
            "Validate and improve operation %d created %d valid thoughts from %d previous thoughts",
            self.id,
            len(
                [
                    thought_list[-1]
                    for thought_list in self.thoughts
                    if thought_list[-1].valid
                ]
            ),
            len(previous_thoughts),
        )


class Generate(Operation):
    """
    用于生成 thoughts 的 operation。
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
        self, num_branches_prompt: int = 1, num_branches_response: int = 1
    ) -> None:
        """
        初始化新的 Generate operation。

        :param num_branches_prompt: 每个提示词应生成的响应数量，会传给 prompter。默认为 1。
        :type num_branches_prompt: int
        :param num_branches_response: LM 应为每个提示词生成的响应数量。默认为 1。
        :type num_branches_response: int
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回与当前 operation 关联的 thoughts。

        :return: 生成的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过基于前驱生成 thoughts 来执行 Generate operation。
        该方法使用前驱的 thought states 向 LM 发送提示来生成 thoughts。
        如果没有前驱，则使用 kwargs 作为基础 state。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # 没有前驱时，使用 kwargs 作为基础 state。
            previous_thoughts = [Thought(state=kwargs)]

        for thought in previous_thoughts:
            base_state = thought.state
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug("Prompt for LM: %s", prompt)
            if self.skip_thought_indices:
                for response_index in range(self.num_branches_response):
                    thought_index = len(self.thoughts)
                    if thought_index in self.skip_thought_indices:
                        self.thoughts.append(
                            _create_skipped_thought(
                                base_state,
                                self,
                                thought_index,
                                self.skip_marker,
                                "generate",
                            )
                        )
                        continue

                    responses, response_metadata = _query_lm_with_metadata(
                        lm, prompt, 1, self, "generate"
                    )
                    for item in response_metadata:
                        item["response_call_index"] = response_index
                        item["requested_num_responses"] = self.num_branches_response
                        item["thought_index"] = thought_index
                    self.logger.debug("Responses from LM: %s", responses)
                    parsed_states = parser.parse_generate_answer(base_state, responses)
                    for parsed_index, new_state in enumerate(parsed_states):
                        new_state = {**base_state, **new_state}
                        metadata = _metadata_for_created_thought(
                            response_metadata, parsed_index
                        )
                        metadata.setdefault("thought_index", len(self.thoughts))
                        self.thoughts.append(Thought(new_state, metadata=metadata))
                        self.logger.debug(
                            "New thought %d created with state %s",
                            self.thoughts[-1].id,
                            self.thoughts[-1].state,
                        )
            else:
                responses, response_metadata = _query_lm_with_metadata(
                    lm, prompt, self.num_branches_response, self, "generate"
                )
                self.logger.debug("Responses from LM: %s", responses)
                for index, new_state in enumerate(
                    parser.parse_generate_answer(base_state, responses)
                ):
                    new_state = {**base_state, **new_state}
                    metadata = _metadata_for_created_thought(response_metadata, index)
                    metadata.setdefault("thought_index", len(self.thoughts))
                    self.thoughts.append(
                        Thought(
                            new_state,
                            metadata=metadata,
                        )
                    )
                    self.logger.debug(
                        "New thought %d created with state %s",
                        self.thoughts[-1].id,
                        self.thoughts[-1].state,
                    )
        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                "Generate operation %d created more thoughts than expected",
                self.id,
            )
        self.logger.info(
            "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
        )


class Improve(Operation):
    """
    用于改进 thoughts 的 operation。
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        """
        初始化新的 Improve operation。
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回当前 operation 改进后的 thoughts。

        :return: 改进后的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过改进前驱中的 thoughts 来执行 Improve operation。
        该方法使用前驱的 thought states 向 LM 发送提示来改进 thoughts。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert len(self.predecessors) > 0, "Needs at least one predecessor"

        for thought in previous_thoughts:
            thought_index = len(self.thoughts)
            if thought_index in self.skip_thought_indices:
                self.thoughts.append(
                    _create_skipped_thought(
                        thought.state,
                        self,
                        thought_index,
                        self.skip_marker,
                        "improve",
                    )
                )
                continue

            improve_prompt = prompter.improve_prompt(**thought.state)
            self.logger.debug("Prompt for LM: %s", improve_prompt)
            responses, improve_metadata = _query_lm_with_metadata(
                lm, improve_prompt, 1, self, "improve"
            )
            for item in improve_metadata:
                item["thought_index"] = thought_index
            self.logger.debug("Responses from LM: %s", responses)
            state_update = parser.parse_improve_answer(thought.state, responses)
            self.thoughts.append(
                Thought(
                    {**thought.state, **state_update},
                    metadata=_metadata_for_created_thought(improve_metadata, 0),
                )
            )

        self.logger.info(
            "Improve operation %d improved %d thoughts", self.id, len(self.thoughts)
        )


class Aggregate(Operation):
    """
    用于聚合 thoughts 的 operation。
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, num_responses: int = 1) -> None:
        """
        初始化新的 Aggregate operation。

        :param num_responses: 聚合时使用的响应数量。默认为 1。
        :type num_responses: int
        """
        super().__init__()
        self.thoughts: List[Thought] = []
        self.num_responses: int = num_responses

    def get_thoughts(self) -> List[Thought]:
        """
        返回当前 operation 聚合后的 thoughts。

        :return: 聚合后的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过聚合前驱中的 thoughts 来执行 Aggregate operation。
        该方法使用前驱的 thought states 向 LM 发送提示来聚合 thoughts。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        """
        assert (
            len(self.predecessors) >= 1
        ), "Aggregate operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            return

        # 按 score 顺序应用。
        base_state: Dict = {}
        for thought in sorted(previous_thoughts, key=lambda thought: thought.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]
        prompt = prompter.aggregation_prompt(previous_thought_states)

        self.logger.debug("Prompt for LM: %s", prompt)

        if self.skip_thought_indices:
            for response_index in range(self.num_responses):
                thought_index = len(self.thoughts)
                if thought_index in self.skip_thought_indices:
                    self.thoughts.append(
                        _create_skipped_thought(
                            base_state,
                            self,
                            thought_index,
                            self.skip_marker,
                            "aggregate",
                        )
                    )
                    continue

                responses, response_metadata = _query_lm_with_metadata(
                    lm, prompt, 1, self, "aggregate"
                )
                for item in response_metadata:
                    item["response_call_index"] = response_index
                    item["requested_num_responses"] = self.num_responses
                    item["thought_index"] = thought_index

                self.logger.debug("Responses from LM: %s", responses)
                parsed = parser.parse_aggregation_answer(
                    previous_thought_states, responses
                )
                if isinstance(parsed, dict):
                    parsed = [parsed]
                for parsed_index, new_state in enumerate(parsed):
                    metadata = _metadata_for_created_thought(
                        response_metadata, parsed_index
                    )
                    metadata.setdefault("thought_index", len(self.thoughts))
                    self.thoughts.append(
                        Thought({**base_state, **new_state}, metadata=metadata)
                    )
        else:
            responses, response_metadata = _query_lm_with_metadata(
                lm, prompt, self.num_responses, self, "aggregate"
            )

            self.logger.debug("Responses from LM: %s", responses)

            parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

            if isinstance(parsed, dict):
                parsed = [parsed]
            for index, new_state in enumerate(parsed):
                metadata = _metadata_for_created_thought(response_metadata, index)
                metadata.setdefault("thought_index", len(self.thoughts))
                self.thoughts.append(
                    Thought(
                        {**base_state, **new_state},
                        metadata=metadata,
                    )
                )

class KeepBestN(Operation):
    """
    根据 score 从前驱中保留最好的 N 个 thoughts 的 operation。
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        """
        初始化新的 KeepBestN operation。

        :param n: 要保留的最大 thoughts 数量。
        :type n: int
        :param higher_is_better: 分数越高是否越好。默认为 True。
        :type higher_is_better: bool
        :raises AssertionError: 如果 `n` 不大于零。
        """
        super().__init__()
        self.n: int = n
        assert self.n > 0, "KeepBestN operation must keep at least one thought"
        self.higher_is_better: bool = higher_is_better
        self.thoughts: List[Thought] = []

    def get_best_n(self) -> List[Thought]:
        """
        根据 score 返回前驱中最好的 N 个 thoughts。

        :return: 最好的 N 个 thoughts 列表。
        :rtype: List[Thought]
        :raises AssertionError: 如果并非所有前驱都已执行。
        :raises AssertionError: 如果并非所有 thoughts 都已打分。
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()
        assert all(
            previous_thought.scored for previous_thought in previous_thoughts
        ), "Not all thoughts have been scored"

        try:
            return sorted(
                previous_thoughts,
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]
        except:
            self.logger.error("Error in KeepBestN operation")
            self.logger.error(
                "Previous operation: %s", [op.id for op in self.predecessors]
            )
            self.logger.error("Previous thoughts: %s", previous_thoughts)
            self.logger.error(
                "Scores: %s", [thought.score for thought in previous_thoughts]
            )
            return sorted(
                [i for i in previous_thoughts if isinstance(i.score, float)],
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]

    def get_thoughts(self) -> List[Thought]:
        """
        返回当前 operation 保留的 thoughts。

        :return: 保留的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过根据 score 从前驱中保留最好的 N 个 thoughts 来执行 KeepBestN operation。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        :raises AssertionError: 如果并非所有前驱都已执行。
        :raises AssertionError: 如果并非所有 thoughts 都已打分。
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepBestN operation must have at least one predecessor"

        self.thoughts = [Thought.from_thought(thought) for thought in self.get_best_n()]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepBestN operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class KeepValid(Operation):
    """
    用于从前驱中保留有效 thoughts 的 operation。
    """

    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        """
        初始化新的 KeepValid operation。
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回当前 operation 保留的 thoughts。

        :return: 保留的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过保留前驱中的有效 thoughts 来执行 KeepValid operation。
        未验证的 thoughts 也会被保留。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepValid operation must have at least one predecessor"

        self.thoughts: List[Thought] = [
            Thought.from_thought(thought)
            for thought in self.get_previous_thoughts()
            if not thought.validated or thought.valid
        ]

        if any(not thought.validated for thought in self.thoughts):
            self.logger.warning(
                "KeepValid operation %d has unvalidated thoughts", self.id
            )

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepValid operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class GroundTruth(Operation):
    """
    使用 ground truth evaluator 判断 thoughts 是否正确解决问题的 operation。
    """

    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(self, ground_truth_evaluator: Callable[[Dict], bool]) -> None:
        """
        初始化新的 GroundTruth operation。

        :param ground_truth_evaluator: 用于判断 thought 是否解决问题的函数。
        :type ground_truth_evaluator: 接收 thought state 并返回布尔值的函数。
        """
        super().__init__()
        self.ground_truth_evaluator: Callable[[Dict], bool] = ground_truth_evaluator
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回与当前 operation 关联的 thoughts。

        :return: 已评估的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过使用 ground truth evaluator 函数评估前驱中的 thoughts 来执行 GroundTruth operation。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        :raises AssertionError: 如果 operation 没有前驱。
        """
        assert (
            len(self.predecessors) >= 1
        ), "GroundTruth operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            try:
                new_thought.solved = self.ground_truth_evaluator(new_thought.state)
            except:
                new_thought.solved = False
            self.thoughts.append(new_thought)

        self.logger.info(
            "GroundTruth operation %d evaluated %d thoughts and %d solved the problem",
            self.id,
            len(self.thoughts),
            len([thought for thought in self.thoughts if thought.solved]),
        )


class Selector(Operation):
    """
    用于从前驱中选择 thoughts 的 operation。
    适用于拆分 thoughts，以便对它们执行不同的后续 operation。
    """

    operation_type: OperationType = OperationType.selector

    def __init__(self, selector: Callable[[List[Thought]], List[Thought]]) -> None:
        """
        初始化新的 Selector operation。

        :param selector: 用于从前驱 thoughts 中选择 thoughts 的函数。
        :type selector: 接收 thoughts 列表并返回 thoughts 列表的函数。
        """
        super().__init__()
        self.selector: Callable[[List[Thought]], List[Thought]] = selector
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回当前 operation 选择的 thoughts。

        :return: 选中的 thoughts 列表。
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        通过使用 selector 函数从前驱中选择 thoughts 来执行 Selector operation。
        如果 Selector 没有前驱，则使用一个以 kwargs 作为 state 的 thought 调用 selector 函数。

        :param lm: 要使用的语言模型。
        :type lm: AbstractLanguageModel
        :param prompter: 用于构造提示词的 prompter。
        :type prompter: Prompter
        :param parser: 用于解析响应的 parser。
        :type parser: Parser
        :param kwargs: 执行时使用的额外参数。
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(kwargs)]

        self.thoughts = [
            Thought.from_thought(thought)
            for thought in self.selector(previous_thoughts)
        ]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s selected", thought.id, thought.state
            )

        self.logger.info(
            "Selector operation %d selected %d thoughts", self.id, len(self.thoughts)
        )
