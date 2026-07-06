"""
Gemini 语言模型模块：实现 Google Gemini 模型的接口。
"""

import backoff
import os
import random
import time
from typing import Any, Dict, List, Union

from google import genai
from google.genai import types as genai_types

from .abstract_language_model import AbstractLanguageModel


class Gemini(AbstractLanguageModel):
    """
    Gemini 语言模型适配类。

    设计目标是与 ChatGPT 类在用法和行为上尽量保持一致：
    - 构造函数签名相同（config_path / model_name / cache）
    - 实现 query() 和 get_response_texts()
    - 累计 prompt_tokens / completion_tokens 并计算 cost
    """

    def __init__(
        self,
        config_path: str = "",
        model_name: str = "gemini-2.5-flash",
        cache: bool = False,
        *,
        ignore_env_api_key: bool = False,
    ) -> None:
        """
        初始化 Gemini 实例。

        :param config_path: 配置文件路径，默认为空（使用默认路径）
        :type config_path: str
        :param model_name: 模型名称，默认为 'gemini-2.5-flash'
        :type model_name: str
        :param cache: 是否缓存响应，默认为 False
        :type cache: bool
        :param ignore_env_api_key: 为 True 时仅用配置文件 api_key（多组 key 并行时避免环境变量覆盖）
        :type ignore_env_api_key: bool
        """
        super().__init__(config_path, model_name, cache)

        # 读取对应 model_name 的配置（结构与 ChatGPT 保持一致）
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]

        # 生成相关参数
        self.temperature: float = self.config["temperature"]
        self.max_tokens: int = self.config["max_tokens"]
        self.stop: Union[str, List[str]] = self.config["stop"]

        # API Key：默认同环境变量；ignore_env_api_key 时仅用配置（多进程多 key 分组）
        if ignore_env_api_key:
            self.api_key = str(self.config.get("api_key", "") or "")
        else:
            self.api_key = os.getenv("GEMINI_API_KEY", self.config.get("api_key", ""))
        if self.api_key == "":
            raise ValueError(
                "GEMINI_API_KEY 未设置，且配置文件中未提供 api_key"
            )

        # 初始化 Gemini 客户端（Google GenAI SDK）
        self.client = genai.Client(api_key=self.api_key)

    def query(self, query: str, num_responses: int = 1) -> Union[List[Any], Any]:
        """
        向 Gemini 模型发送查询请求。

        对外语义与 ChatGPT.query 对齐：
        - num_responses == 1 时返回单个响应对象
        - num_responses > 1 时返回响应对象列表

        :param query: 发送给模型的查询内容
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: Gemini 模型的响应
        :rtype: Union[List[Any], Any]
        """
        # 检查缓存
        if self.cache and query in getattr(self, "response_cache", {}):
            return self.response_cache[query]

        messages = [{"role": "user", "content": query}]

        if num_responses == 1:
            response = self.chat(messages)
        else:
            # Gemini API 当前单次只支持一个候选，通过多次调用模拟 num_responses
            response_list: List[Any] = []
            remaining = num_responses
            total_num_attempts = num_responses
            while remaining > 0 and total_num_attempts > 0:
                try:
                    res = self.chat(messages)
                    response_list.append(res)
                    remaining -= 1
                except Exception as e:
                    self.logger.warning(
                        f"Gemini 请求出错: {e}，短暂等待后重试"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1
            response = response_list

        # 缓存响应
        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, Exception, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> Any:
        """
        与 Gemini 进行一次对话调用。

        由于 Gemini SDK 当前单次调用仅返回 1 个候选，num_responses 参数仅为兼容签名；
        多样本的逻辑由 query() 层通过多次调用本方法来模拟。

        :param messages: 消息列表，每个消息包含 role 和 content
        :type messages: List[Dict]
        :param num_responses: 期望的响应数量（当前版本仅支持 1）
        :type num_responses: int
        :return: Gemini 模型的响应
        :rtype: Any
        """
        # 将所有 message 文本拼接成一个 prompt
        contents: List[str] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                contents.append(content)
        prompt = "\n".join(contents)

        # 生成配置：只设置已知兼容的字段
        generation_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        start_time = self._now()
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=generation_config,
        )
        latency = self._now() - start_time

        # 统计 token 使用情况并计算费用
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

        self._calculate_cost()
        setattr(response, "_got_latency_seconds", latency)

        self.logger.info(
            "Gemini 响应: %s\n当前累计费用: $%.4f",
            response,
            self.cost,
        )
        return response

    def get_response_texts(self, query_response: Union[List[Any], Any]) -> List[str]:
        """
        从 Gemini 响应对象（或其列表）中提取纯文本列表。

        与 ChatGPT.get_response_texts 一样：
        - 如果是单个响应对象，则包装成列表
        - 返回所有响应的文本列表

        :param query_response: Gemini 模型的响应
        :type query_response: Union[List[Any], Any]
        :return: 响应文本列表
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]

        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for response in query_response:
            usage = getattr(response, "usage_metadata", None)
            response_metadata: Dict[str, Any] = {
                # provider/model: 标记响应来自原生 Gemini SDK 以及实际模型 id。
                "provider": "gemini",
                "model": self.model_id,
                # 原生 Gemini SDK 当前未稳定暴露 token logprobs，因此不能直接计算熵。
                "has_logprobs": False,
                # latency_seconds: 本次 Gemini API 调用耗时，单位秒。
                "latency_seconds": getattr(response, "_got_latency_seconds", None),
                "usage": {
                    # prompt_tokens: 输入 prompt 消耗的 token 数。
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    # completion_tokens: 模型输出消耗的 token 数。
                    "completion_tokens": getattr(usage, "candidates_token_count", None),
                    # total_tokens: prompt + completion 的总 token 数。
                    "total_tokens": getattr(usage, "total_token_count", None),
                },
            }
            # 优先使用 SDK 提供的便捷属性 .text
            text = getattr(response, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
                metadata.append(response_metadata)
                continue

            # 兜底：从 candidates[0].content.parts[0].text 中取
            candidates = getattr(response, "candidates", None)
            if not candidates:
                continue
            candidate0 = candidates[0]
            content = getattr(candidate0, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            first_part = parts[0]
            part_text = getattr(first_part, "text", None)
            if isinstance(part_text, str) and part_text:
                texts.append(part_text)
                metadata.append(response_metadata)

        self._set_last_response_metadata(metadata)
        return texts
