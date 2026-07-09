# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Robert Gerstenberger, Nils Blach

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Parser(ABC):
    """
    定义所有 parser 接口的抽象基类。
    Parser 用于解析语言模型返回的响应。
    """

    @abstractmethod
    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        解析语言模型对 aggregation prompt 的响应。

        :param states: 用于生成提示词的 thought states。
        :type states: List[Dict]
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: 解析语言模型响应后得到的新 thought states。
        :rtype: Union[Dict, List[Dict]]
        """
        pass

    @abstractmethod
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        解析语言模型对 improve prompt 的响应。

        :param state: 用于生成提示词的 thought state。
        :type state: Dict
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: 解析语言模型响应后得到的新 thought state。
        :rtype: Dict
        """
        pass

    @abstractmethod
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        解析语言模型对 generate prompt 的响应。

        :param state: 用于生成提示词的 thought state。
        :type state: Dict
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: 解析语言模型响应后得到的新 thought states。
        :rtype: List[Dict]
        """
        pass

    @abstractmethod
    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        解析语言模型对 validation prompt 的响应。

        :param state: 用于生成提示词的 thought state。
        :type state: Dict
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: thought state 是否有效。
        :rtype: bool
        """
        pass

    @abstractmethod
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        解析语言模型对 score prompt 的响应。

        :param states: 用于生成提示词的 thought states。
        :type states: List[Dict]
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: thought states 的分数。
        :rtype: List[float]
        """
        pass
