# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

import backoff
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    ChatGPT 类使用提供的配置处理与 OpenAI 模型的交互。

    该类继承 AbstractLanguageModel，并实现其抽象方法。
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False
    ) -> None:
        """
        使用配置、模型信息和缓存选项初始化 ChatGPT 实例。

        :param config_path: 配置文件路径。默认为 ""。
        :type config_path: str
        :param model_name: 模型名称，默认为 'chatgpt'。用于选择正确配置。
        :type model_name: str
        :param cache: 是否缓存响应。默认为 False。
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # model_id 是 chatgpt 使用的模型 id，例如 gpt-4、gpt-3.5-turbo 等。
        self.model_id: str = self.config["model_id"]
        # prompt_token_cost 和 response_token_cost 分别是每 1000 个 prompt token 和 response token 的成本。
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # temperature 表示模型输出的随机性。
        self.temperature: float = self.config["temperature"]
        # 聊天补全中最多生成的 token 数量。
        self.max_tokens: int = self.config["max_tokens"]
        # stop sequence 是让模型停止生成的 token 序列，模型不会生成该序列本身。
        self.stop: Union[str, List[str]] = self.config["stop"]
        # collect_logprobs: 是否向 API 请求 token 级 logprob 信息。
        # 只有开启后，后续 thought_metadata 才可能包含 entropy/logprob 字段。
        self.collect_logprobs: bool = bool(self.config.get("collect_logprobs", False))
        # top_logprobs: 每个生成位置返回概率最高的几个候选 token。
        # 例如 5 表示返回 top-5 候选，用这些候选概率近似计算 token entropy。
        self.top_logprobs: int = int(self.config.get("top_logprobs", 5))
        # top_logprobs_normalized_mass: 可选归一化口径。
        # 例如设为 0.99，表示把 API 返回的 top-k 候选概率重标定为覆盖 0.99 概率质量。
        self.top_logprobs_normalized_mass: float = float(
            self.config.get("top_logprobs_normalized_mass", 0.0) or 0.0
        )
        # account organization 是 chatgpt 使用的组织。
        self.organization: str = self.config["organization"]
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION is not set")
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")
        # 初始化 OpenAI Client。
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        查询 OpenAI 模型以获取响应。

        :param query: 发送给语言模型的查询。
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1。
        :type num_responses: int
        :return: OpenAI 模型返回的响应。
        :rtype: Dict
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in chatgpt: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        向 OpenAI 模型发送聊天消息并获取模型响应。
        遇到 OpenAI 错误时使用 backoff 重试。

        :param messages: 聊天消息字典列表。
        :type messages: List[Dict]
        :param num_responses: 期望的响应数量，默认为 1。
        :type num_responses: int
        :return: OpenAI 模型的响应。
        :rtype: ChatCompletion
        """
        create_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": num_responses,
            "stop": self.stop,
        }
        if self.collect_logprobs:
            # logprobs=True: 请求实际生成 token 的 log 概率。
            create_kwargs["logprobs"] = True
            # top_logprobs=k: 请求每个位置 top-k 候选 token 的 log 概率。
            # 完整熵需要全词表概率，但 API 通常只返回 top-k，因此这里得到的是近似熵。
            create_kwargs["top_logprobs"] = self.top_logprobs

        start_time = self._now()
        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except OpenAIError:
            if not self.collect_logprobs:
                raise
            self.logger.warning(
                "Model/provider rejected logprobs request; retrying without logprobs"
            )
            create_kwargs.pop("logprobs", None)
            create_kwargs.pop("top_logprobs", None)
            response = self.client.chat.completions.create(**create_kwargs)
        latency = self._now() - start_time

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        self._calculate_cost()
        setattr(response, "_got_latency_seconds", latency)
        self._record_model_call(latency)
        self.logger.info(
            f"This is the response from chatgpt: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        从查询响应中提取响应文本。

        :param query_response: OpenAI 模型返回的响应字典或字典列表。
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: 响应字符串列表。
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]
        metadata = self._openai_chat_metadata(query_response, "openai")
        for item, response in zip(metadata, query_response):
            item["latency_seconds"] = getattr(response, "_got_latency_seconds", None)
        self._set_last_response_metadata(metadata)
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
