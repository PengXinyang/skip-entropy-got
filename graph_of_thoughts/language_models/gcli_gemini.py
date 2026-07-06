"""
GCLI Gemini 语言模型模块：通过 GCLI 代理访问 Gemini 模型。

GCLI 是一个提供 OpenAI 兼容 API 的 Gemini 代理服务，
允许使用 OpenAI SDK 调用 Gemini 模型。
"""

import os
import random
import time
from typing import Dict, List, Union

import backoff
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class GCLIGemini(AbstractLanguageModel):
    """
    使用 GCLI 代理访问 Gemini 模型（如 gemini-2.5-flash / gemini-2.5-pro）的适配类。

    设计目标：
    - 与 ChatGPT / Gemini 的用法基本一致（构造函数 / query / get_response_texts）
    - 复用现有 config.json 中的 Gemini 模型配置
    - 通过 OpenAI 兼容的 /v1/chat/completions 接口调用 GCLI 代理
    """

    def __init__(
        self,
        config_path: str = "",
        model_name: str = "gemini-2.5-flash",
        cache: bool = False,
    ) -> None:
        """
        初始化 GCLIGemini 实例。

        :param config_path: 配置文件路径（默认与其它模型共用的 config.json）
        :type config_path: str
        :param model_name: 使用的模型名称，如 "gemini-2.5-flash" 或 "gemini-2.5-pro"
        :type model_name: str
        :param cache: 是否开启本地缓存，默认为 False
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)

        # 复用 config.json 中对应 model_name 的配置
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]

        self.temperature: float = self.config["temperature"]
        self.max_tokens: int = self.config["max_tokens"]
        self.stop: Union[str, List[str]] = self.config["stop"]
        # collect_logprobs: 是否请求 token 级 logprob；代理支持时可用于计算节点熵。
        self.collect_logprobs: bool = bool(self.config.get("collect_logprobs", False))
        # top_logprobs: 每个生成位置返回概率最高的 k 个候选 token，用于近似熵。
        self.top_logprobs: int = int(self.config.get("top_logprobs", 5))

        # GCLI 代理相关配置
        # 优先使用环境变量，否则使用 config.json 中的 api_key / base_url
        self.api_key: str = os.getenv("GCLI_API_KEY", self.config.get("api_key", ""))
        if self.api_key == "":
            raise ValueError(
                "GCLI_API_KEY 未设置，且配置文件中未提供 api_key"
            )

        # OpenAI 兼容接口通常是 <base_url>/v1/chat/completions
        # 默认使用 GCLI 公共地址，可在配置中覆盖
        self.base_url: str = self.config.get(
            "base_url",
            "https://gcli.ggchan.dev",
        )

        # 使用 OpenAI SDK，将 base_url 指向 GCLI 代理
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        向 GCLI 代理发送查询请求。

        对外接口与 ChatGPT.query 保持一致：
        - num_responses == 1 时返回单个 ChatCompletion
        - num_responses > 1 时返回 ChatCompletion 列表

        :param query: 发送给模型的查询内容
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: 模型的响应
        :rtype: Union[List[ChatCompletion], ChatCompletion]
        """
        # 检查缓存
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}])
        else:
            # GCLI（OpenAI 兼容）常见限制：同一请求仅允许 n=1。
            # 通过多次 n=1 调用模拟多样本，避免 400: n_limit_exceeded。
            response_list: List[ChatCompletion] = []
            remaining = num_responses
            total_num_attempts = num_responses
            while remaining > 0 and total_num_attempts > 0:
                try:
                    res = self.chat([{"role": "user", "content": query}])
                    response_list.append(res)
                    remaining -= 1
                except Exception as e:
                    self.logger.warning(
                        f"GCLIGemini 请求出错: {e}，稍后重试 (remaining={remaining})"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1
            response = response_list

        # 缓存响应
        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        通过 GCLI 代理向 Gemini 模型发起一次 /v1/chat/completions 调用。

        GCLI 侧多数 API Key 仅允许单次请求 ``n=1``；多样本由 ``query()`` 多次调用本方法模拟。
        ``num_responses`` 参数仅为与其它 LM 类签名兼容，请求体中固定 ``n=1``。

        :param messages: 消息列表，每个消息是包含 role 和 content 的字典
        :type messages: List[Dict]
        :param num_responses: 兼容签名，实际不使用
        :type num_responses: int
        :return: 模型的响应
        :rtype: ChatCompletion
        """
        create_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stop": self.stop,
        }
        if self.collect_logprobs:
            # logprobs=True 返回实际生成 token 的 log 概率。
            create_kwargs["logprobs"] = True
            # top_logprobs=k 返回 top-k 候选 token 的 log 概率，用于近似 token entropy。
            create_kwargs["top_logprobs"] = self.top_logprobs

        start_time = self._now()
        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except OpenAIError:
            if not self.collect_logprobs:
                raise
            self.logger.warning(
                "GCLI rejected logprobs request; retrying without logprobs"
            )
            create_kwargs.pop("logprobs", None)
            create_kwargs.pop("top_logprobs", None)
            response = self.client.chat.completions.create(**create_kwargs)
        latency = self._now() - start_time

        # 统计 token 使用和费用（GCLI 为 OpenAI 兼容接口时通常会返回 usage 字段）
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            self._calculate_cost()
        setattr(response, "_got_latency_seconds", latency)

        self.logger.info(
            f"GCLIGemini 响应: {response}"
            f"\n当前累计费用: ${self.cost:.4f}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        从 ChatCompletion 响应中提取纯文本。

        与 ChatGPT.get_response_texts 行为保持一致。

        :param query_response: 模型的响应
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: 响应文本列表
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]
        metadata = self._openai_chat_metadata(query_response, "gcli_gemini")
        for item, response in zip(metadata, query_response):
            item["latency_seconds"] = getattr(response, "_got_latency_seconds", None)
        self._set_last_response_metadata(metadata)
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
