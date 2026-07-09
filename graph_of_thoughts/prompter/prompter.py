# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Robert Gerstenberger, Nils Blach

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class Prompter(ABC):
    """
    定义所有 prompter 接口的抽象基类。
    Prompter 用于为语言模型生成提示词。
    """

    @abstractmethod
    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        为语言模型生成 aggregation prompt。

        :param state_dicts: 应该被聚合的 thought states。
        :type state_dicts: List[Dict]
        :param kwargs: 额外关键字参数。
        :return: aggregation prompt。
        :rtype: str
        """
        pass

    @abstractmethod
    def improve_prompt(self, **kwargs) -> str:
        """
        为语言模型生成 improve prompt。
        thought state 会被展开，以允许额外关键字参数，
        并允许具体实现显式声明所需参数。

        :param kwargs: 额外关键字参数。
        :return: improve prompt。
        :rtype: str
        """
        pass

    @abstractmethod
    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        """
        为语言模型生成 generate prompt。
        thought state 会被展开，以允许额外关键字参数，
        并允许具体实现显式声明所需参数。

        :param num_branches: 提示词要求 LM 生成的响应数量。
        :type num_branches: int
        :param kwargs: 额外关键字参数。
        :return: generate prompt。
        :rtype: str
        """
        pass

    @abstractmethod
    def validation_prompt(self, **kwargs) -> str:
        """
        为语言模型生成 validation prompt。
        thought state 会被展开，以允许额外关键字参数，
        并允许具体实现显式声明所需参数。

        :param kwargs: 额外关键字参数。
        :return: validation prompt。
        :rtype: str
        """
        pass

    @abstractmethod
    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        为语言模型生成 score prompt。

        :param state_dicts: 应该被打分的 thought states；
                            如果超过一个，则应该一起打分。
        :type state_dicts: List[Dict]
        :param kwargs: 额外关键字参数。
        :return: score prompt。
        :rtype: str
        """
        pass
