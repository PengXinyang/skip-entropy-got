# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Ales Kubicek

import os
import torch
import time
from typing import List, Dict, Union
from .abstract_language_model import AbstractLanguageModel


class Llama2HF(AbstractLanguageModel):
    """
    通过 HuggingFace 库使用 LLaMA 2 模型的接口。
    """

    def __init__(
        self, config_path: str = "", model_name: str = "llama7b-hf", cache: bool = False
    ) -> None:
        """
        使用配置、模型信息和缓存选项初始化 Llama2HF 实例。

        :param config_path: 配置文件路径。默认为空字符串。
        :type config_path: str
        :param model_name: 指定 LLaMA 模型变体名称。默认为 "llama7b-hf"。
                           用于选择正确配置。
        :type model_name: str
        :param cache: 是否缓存响应。默认为 False。
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # 所用模型的详细 id。
        self.model_id: str = self.config["model_id"]
        # 每 1000 个 token 的成本。
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # temperature 表示模型输出的随机性。
        self.temperature: float = self.config["temperature"]
        # Top K 采样。
        self.top_k: int = self.config["top_k"]
        # 聊天补全中最多生成的 token 数量。
        self.max_tokens: int = self.config["max_tokens"]

        # 注意：必须在导入 transformers 前完成。
        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]
        import transformers

        hf_model_id = f"meta-llama/{self.model_id}"
        model_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        torch.no_grad()

        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-generation"
        )

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        查询 LLaMA 2 模型以获取响应。

        :param query: 发送给语言模型的查询。
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1。
        :type num_responses: int
        :return: LLaMA 2 模型返回的响应。
        :rtype: List[Dict]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]
        sequences = []
        query = f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {query} [/INST]"
        for _ in range(num_responses):
            start_time = time.perf_counter()
            sequences.extend(
                self.generate_text(
                    query,
                    do_sample=True,
                    top_k=self.top_k,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_tokens,
                )
            )
            self._record_model_call(time.perf_counter() - start_time)
        response = [
            {"generated_text": sequence["generated_text"][len(query) :].strip()}
            for sequence in sequences
        ]
        if self.cache:
            self.response_cache[query] = response
        return response

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        从查询响应中提取响应文本。

        :param query_responses: `query` 方法生成的响应字典列表。
        :type query_responses: List[Dict]
        :return: 响应字符串列表。
        :rtype: List[str]
        """
        texts = [query_response["generated_text"] for query_response in query_responses]
        self._set_last_response_metadata(
            [
                {
                    "provider": "huggingface",
                    "model": self.model_id,
                    "has_logprobs": False,
                    "usage": {
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                    },
                }
                for _ in texts
            ]
        )
        return texts
