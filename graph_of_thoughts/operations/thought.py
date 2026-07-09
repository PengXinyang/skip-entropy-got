# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

from __future__ import annotations
import logging
from typing import Iterator, Dict, Optional, Any
import itertools


class Thought:
    """
    表示由 parser 构造的 LLM thought，包含其 state 和各种标记。
    """

    _ids: Iterator[int] = itertools.count(0)

    def __init__(
        self,
        state: Optional[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        使用 state 和各种默认标记初始化新的 Thought 实例。

        :param state: thought 的 state。默认为 None。
        :type state: Optional[Dict]
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Thought._ids)
        # state: 业务状态，例如 current/result/phase 等，由各任务 parser 解析得到。
        self.state: Dict = state
        # metadata: 观测信息，例如生成该 Thought 的模型、prompt、response、熵、token 用量、延迟等。
        self.metadata: Dict[str, Any] = metadata or {}
        self._score: float = 0.0
        self._valid: bool = False
        self._solved: bool = False
        self.scored: bool = False
        self.validated: bool = False
        self.compared_to_ground_truth: bool = False

    @staticmethod
    def from_thought(thought: Thought) -> Thought:
        """
        基于已有 thought 创建一个新的 thought。

        :param thought: 要克隆的 Thought 实例。
        :return: 从输入 thought 复制属性得到的新 Thought 实例。
        """
        new_thought = Thought(thought.state)
        # 复制 Thought 时保留观测信息，便于 Score/KeepBestN/GroundTruth 后仍能追溯来源。
        new_thought.metadata = dict(thought.metadata)
        new_thought.score = thought.score
        new_thought.valid = thought.valid
        new_thought.solved = thought.solved
        new_thought.scored = thought.scored
        new_thought.validated = thought.validated
        new_thought.compared_to_ground_truth = thought.compared_to_ground_truth
        return new_thought

    @property
    def valid(self) -> bool:
        """
        返回 thought 的有效性。

        :return: thought 的有效性。
        :rtype: bool
        """
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        """
        设置 thought 的有效性和 validated 标记。

        :param valid: thought 的有效性。
        :type valid: bool
        """
        self.validated = True
        self._valid = valid

    @property
    def score(self) -> float:
        """
        返回 thought 的分数。

        :return: thought 的分数。
        :rtype: float
        """
        return self._score

    @score.setter
    def score(self, new_score: float) -> None:
        """
        设置 thought 的分数和 scored 标记。

        :param new_score: thought 的分数。
        :type new_score: float
        """
        self.scored = True
        self._score = new_score

    @property
    def solved(self) -> bool:
        """
        返回 thought 的 solved 标记。

        :return: thought 的 solved 标记。
        :rtype: bool
        """
        return self._solved

    @solved.setter
    def solved(self, solved: bool) -> None:
        """
        设置 thought 的 solved 标记和 compared_to_ground_truth 标记。

        :param solved: thought 是否包含问题的解。
        :type solved: bool
        """
        self.compared_to_ground_truth = True
        self._solved = solved
