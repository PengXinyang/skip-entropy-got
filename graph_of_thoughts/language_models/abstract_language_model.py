# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import json
import os
import logging
import math
import time


class AbstractLanguageModel(ABC):
    """
    定义所有语言模型接口的抽象基类。
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False
    ) -> None:
        """
        使用配置、模型信息和缓存选项初始化 AbstractLanguageModel 实例。

        :param config_path: 配置文件路径。默认为 ""。
        :type config_path: str
        :param model_name: 语言模型名称。默认为 ""。
        :type model_name: str
        :param cache: 是否缓存响应。默认为 False。
        :type cache: bool
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.model_name: str = model_name
        self.cache = cache
        if self.cache:
            self.response_cache: Dict[str, List[Any]] = {}
        self.load_config(config_path)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0
        # api_calls: 实际向模型后端发起请求的次数。
        # 当 num_responses 被拆成多次 num_responses=1 时，这个值会真实反映调用次数。
        self.api_calls: int = 0
        # total_latency_seconds: 所有模型请求累计耗时，单位秒，用于比较时间成本。
        self.total_latency_seconds: float = 0.0
        # 保存最近一次 get_response_texts() 提取出的观测信息。
        # operations.py 会在创建 Thought 时消费它，并写入 thought.metadata。
        self.last_response_metadata: List[Dict[str, Any]] = []

    def _now(self) -> float:
        """
        返回当前墙钟时间，用于延迟测量。
        """
        return time.perf_counter()

    def _calculate_cost(self) -> None:
        """
        当具体模型提供 OpenAI 风格的 token 价格时，根据累计 token 计数重新计算累计成本。
        """
        prompt_token_cost = getattr(self, "prompt_token_cost", 0.0)
        response_token_cost = getattr(self, "response_token_cost", 0.0)
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            prompt_token_cost * prompt_tokens_k
            + response_token_cost * completion_tokens_k
        )

    def _record_model_call(self, latency_seconds: float = 0.0) -> None:
        """
        记录一次真实模型调用。API 模型和本地模型都通过这个字段统一统计调用次数。
        """
        self.api_calls += 1
        if latency_seconds is not None:
            self.total_latency_seconds += float(latency_seconds)

    def _entropy_from_top_logprobs(self, top_logprobs: List[Any]) -> Union[float, None]:
        """
        根据 provider 返回的 top-logprobs 列表近似计算 token 熵。
        如果 provider 只返回 top-k token，则这是下界风格的估计。
        """
        # top_logprobs 是模型在某个生成位置上返回的“概率最高的 k 个候选 token”
        # 及其 log 概率。完整熵需要整个词表分布；这里只能用 top-k 概率做近似。
        if not top_logprobs:
            return None

        probs = []
        for item in top_logprobs:
            logprob = getattr(item, "logprob", None)
            if logprob is None and isinstance(item, dict):
                logprob = item.get("logprob")
            if logprob is None:
                continue
            probs.append(math.exp(float(logprob)))

        if not probs:
            return None

        return -sum(prob * math.log2(prob) for prob in probs if prob > 0.0)

    def _normalized_entropy_from_top_logprobs(
        self, top_logprobs: List[Any], target_mass: float = 0.99
    ) -> Union[float, None]:
        """
        将 API 返回的 top-k 概率重新缩放到指定概率质量后计算熵。

        例如 target_mass=0.99 表示：假设 top-k token 覆盖了 0.99 的概率质量，
        先把 top-k 内部相对概率归一化，再整体乘以 0.99。
        这不是完整词表熵，只是为了让不同 API 的 top-k 近似口径更可比。
        """
        if target_mass <= 0.0:
            return None
        if not top_logprobs:
            return None

        probs = []
        for item in top_logprobs:
            logprob = getattr(item, "logprob", None)
            if logprob is None and isinstance(item, dict):
                logprob = item.get("logprob")
            if logprob is None:
                continue
            probs.append(math.exp(float(logprob)))

        prob_sum = sum(probs)
        if prob_sum <= 0.0:
            return None

        scaled_probs = [(prob / prob_sum) * target_mass for prob in probs]
        return -sum(prob * math.log2(prob) for prob in scaled_probs if prob > 0.0)

    def _extract_choice_logprob_metadata(self, choice: Any) -> Dict[str, Any]:
        """
        从 OpenAI 兼容的 choices 中提取 token 级可观测字段。
        不同 provider 的 logprobs schema 略有差异，因此这里使用防御性的属性和字典访问。
        """
        choice_logprobs = getattr(choice, "logprobs", None)
        if choice_logprobs is None:
            return {}

        content_logprobs = getattr(choice_logprobs, "content", None)
        if content_logprobs is None and isinstance(choice_logprobs, dict):
            content_logprobs = choice_logprobs.get("content")
        if not content_logprobs:
            return {}

        token_logprobs = []
        token_entropies = []
        normalized_token_entropies = []
        tokens = []
        for token_info in content_logprobs:
            token = getattr(token_info, "token", None)
            logprob = getattr(token_info, "logprob", None)
            top_logprobs = getattr(token_info, "top_logprobs", None)
            if isinstance(token_info, dict):
                token = token_info.get("token", token)
                logprob = token_info.get("logprob", logprob)
                top_logprobs = token_info.get("top_logprobs", top_logprobs)

            if token is not None:
                tokens.append(token)
            if logprob is not None:
                token_logprobs.append(float(logprob))

            entropy = self._entropy_from_top_logprobs(top_logprobs or [])
            if entropy is not None:
                token_entropies.append(entropy)

            normalized_entropy = self._normalized_entropy_from_top_logprobs(
                top_logprobs or [],
                float(
                    getattr(self, "top_logprobs_normalized_mass", 0.0)
                    or 0.0
                ),
            )
            if normalized_entropy is not None:
                normalized_token_entropies.append(normalized_entropy)

        metadata: Dict[str, Any] = {
            # tokens: 实际生成出来的 token 序列，用于定位每个 token 的不确定性。
            "tokens": tokens,
            # token_logprobs: 实际生成 token 的 log 概率；越接近 0 说明模型越确信。
            "token_logprobs": token_logprobs,
            # num_observed_tokens: API 返回了 logprob 信息的 token 数。
            "num_observed_tokens": len(content_logprobs),
        }
        if token_logprobs:
            # avg_neg_logprob_bits: 平均负 log 概率，单位转成 bit。
            # 没有 top_logprobs 时，也可用它作为“生成不确定性”的替代指标。
            metadata["avg_neg_logprob_bits"] = (
                -sum(token_logprobs) / len(token_logprobs) / math.log(2)
            )
        if token_entropies:
            # token_entropies_bits: 每个 token 位置基于 top_logprobs 估计的熵。
            metadata["token_entropies_bits"] = token_entropies
            # sum_entropy_bits: 一个 response/节点的总熵，可对应论文里的 step entropy。
            metadata["sum_entropy_bits"] = sum(token_entropies)
            # avg_entropy_bits: 长度归一化后的平均熵，便于比较不同长度节点。
            metadata["avg_entropy_bits"] = sum(token_entropies) / len(token_entropies)
            # entropy_estimate: 标记当前熵是用 top-k 候选概率估计的，不是完整词表熵。
            metadata["entropy_estimate"] = "top_logprobs"
        if normalized_token_entropies:
            # normalized_token_entropies_bits: 假设 top-k 概率质量为固定值后的归一化熵。
            # 例如 top_logprobs_normalized_mass=0.99 时，top-k 内部相对概率会被缩放到 0.99。
            metadata["normalized_token_entropies_bits"] = normalized_token_entropies
            metadata["normalized_sum_entropy_bits"] = sum(normalized_token_entropies)
            metadata["normalized_avg_entropy_bits"] = (
                sum(normalized_token_entropies) / len(normalized_token_entropies)
            )
            metadata["normalized_entropy_mass"] = getattr(
                self, "top_logprobs_normalized_mass", None
            )
        return metadata

    def _set_last_response_metadata(
        self,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        保存与 get_response_texts() 返回列表对齐的元数据。
        """
        self.last_response_metadata = metadata

    def consume_last_response_metadata(
        self, expected_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        返回最近一次 get_response_texts() 调用产生的元数据，
        并按调用方正在处理的响应文本数量进行填充或截断。
        """
        metadata = list(self.last_response_metadata or [])
        if expected_count > 0:
            if len(metadata) < expected_count:
                metadata.extend({} for _ in range(expected_count - len(metadata)))
            elif len(metadata) > expected_count:
                metadata = metadata[:expected_count]
        self.last_response_metadata = []
        return metadata

    def _openai_chat_metadata(
        self,
        query_response: Union[List[Any], Any],
        provider: str,
    ) -> List[Dict[str, Any]]:
        """
        为 OpenAI 兼容的 ChatCompletion 对象构造逐 choice 元数据。
        """
        if not isinstance(query_response, list):
            query_response = [query_response]

        metadata: List[Dict[str, Any]] = []
        for response in query_response:
            usage = getattr(response, "usage", None)
            latency = getattr(response, "_got_latency_seconds", None)
            response_usage = {
                # prompt_tokens: 输入 prompt 消耗的 token 数。
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                # completion_tokens: 模型输出消耗的 token 数。
                "completion_tokens": getattr(usage, "completion_tokens", None),
                # total_tokens: prompt + completion 的总 token 数。
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            for choice in getattr(response, "choices", []) or []:
                choice_metadata: Dict[str, Any] = {
                    # provider: 当前响应来自哪个后端，例如 openai/deepseek/gcli_gemini。
                    "provider": provider,
                    # model: 实际调用的模型 id。
                    "model": getattr(self, "model_id", self.model_name),
                    # usage: token 用量统计，用于成本和压缩率分析。
                    "usage": response_usage,
                    # finish_reason: 模型停止原因，例如 stop/length/content_filter。
                    "finish_reason": getattr(choice, "finish_reason", None),
                    # has_logprobs: 是否真的拿到了 logprobs；没有则无法直接算熵。
                    "has_logprobs": getattr(choice, "logprobs", None) is not None,
                    # latency_seconds: 本次 API 调用耗时，单位秒。
                    "latency_seconds": latency,
                }
                choice_metadata.update(self._extract_choice_logprob_metadata(choice))
                metadata.append(choice_metadata)
        return metadata

    def load_config(self, path: str) -> None:
        """
        从指定路径加载配置。

        :param path: 配置文件路径。如果提供空路径，则默认使用当前目录中的 `config.json`。
        :type path: str
        """
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "config.json")

        with open(path, "r") as f:
            self.config = json.load(f)

        self.logger.debug(f"Loaded config from {path} for {self.model_name}")

    def clear_cache(self) -> None:
        """
        清空响应缓存。
        """
        self.response_cache.clear()

    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        查询语言模型的抽象方法。

        :param query: 发送给语言模型的查询。
        :type query: str
        :param num_responses: 期望的响应数量。
        :type num_responses: int
        :return: 语言模型的响应。
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        从语言模型响应中提取响应文本的抽象方法。

        :param query_responses: 语言模型返回的响应。
        :type query_responses: Union[List[Any], Any]
        :return: 文本响应列表。
        :rtype: List[str]
        """
        pass
