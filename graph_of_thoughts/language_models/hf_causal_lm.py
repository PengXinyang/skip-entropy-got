"""
通用 HuggingFace CausalLM 后端。

该后端用于本地模型实验，重点是拿到每个生成 token 的完整词表 logits，
从而计算完整词表 token entropy，而不是 API top-k 近似熵。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch

from .abstract_language_model import AbstractLanguageModel


class HFCausalLM(AbstractLanguageModel):
    """
    使用 transformers AutoModelForCausalLM 的本地模型适配器。

    适合 DeepSeek-R1-Distill-Qwen-32B、Qwen、Llama 等 causal language model。
    """

    def __init__(
        self, config_path: str = "", model_name: str = "deepseek-32b-hf", cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict[str, Any] = self.config[model_name]

        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = float(self.config.get("prompt_token_cost", 0.0))
        self.response_token_cost: float = float(
            self.config.get("response_token_cost", 0.0)
        )
        self.temperature: float = float(self.config.get("temperature", 0.6))
        self.top_p: float = float(self.config.get("top_p", 0.95))
        self.top_k: int = int(self.config.get("top_k", 0) or 0)
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 1024))
        self.do_sample: bool = bool(self.config.get("do_sample", True))
        self.system_prompt: str = str(
            self.config.get(
                "system_prompt",
                "You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.",
            )
        )
        self.torch_dtype: str = str(self.config.get("torch_dtype", "bfloat16"))
        self.device_map: str = str(self.config.get("device_map", "auto"))
        self.load_in_4bit: bool = bool(self.config.get("load_in_4bit", True))

        import transformers

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(
                    self.config.get("bnb_4bit_quant_type", "nf4")
                ),
                bnb_4bit_use_double_quant=bool(
                    self.config.get("bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=dtype,
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=self.config.get("cache_dir"),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": self.device_map,
            "torch_dtype": dtype,
            "cache_dir": self.config.get("cache_dir"),
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self.model.eval()

    def _build_prompt(self, query: str) -> str:
        """
        优先使用 tokenizer 的 chat_template；没有模板时退化为普通 user prompt。
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.system_prompt}\n\nUser: {query}\nAssistant:"

    def _entropy_metadata_from_scores(
        self,
        scores: List[torch.Tensor],
        generated_token_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        用完整词表 logits 计算每个生成 token 的熵。

        scores[t] 是第 t 个生成位置对整个词表的 logits，形状为 [1, vocab_size]。
        熵公式：H_t = -sum_v p(v|context) log2 p(v|context)。
        """
        token_logprobs: List[float] = []
        token_entropies_bits: List[float] = []

        for index, logits in enumerate(scores):
            logits = logits[0].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy_bits = -torch.sum(probs * log_probs).item() / torch.log(
                torch.tensor(2.0)
            ).item()
            token_entropies_bits.append(float(entropy_bits))

            token_id = int(generated_token_ids[index].item())
            token_logprobs.append(float(log_probs[token_id].item()))

        tokens = self.tokenizer.convert_ids_to_tokens(
            generated_token_ids.detach().cpu().tolist()
        )
        metadata: Dict[str, Any] = {
            "provider": "huggingface",
            "model": self.model_id,
            "has_logprobs": True,
            "entropy_estimate": "full_vocab",
            "tokens": tokens,
            "token_ids": generated_token_ids.detach().cpu().tolist(),
            "token_logprobs": token_logprobs,
            "num_observed_tokens": len(token_entropies_bits),
            "vocab_size": int(getattr(self.model.config, "vocab_size", 0) or 0),
            "token_entropies_bits": token_entropies_bits,
            "sum_entropy_bits": sum(token_entropies_bits),
            "avg_entropy_bits": (
                sum(token_entropies_bits) / len(token_entropies_bits)
                if token_entropies_bits
                else None
            ),
        }
        if token_logprobs:
            metadata["avg_neg_logprob_bits"] = (
                -sum(token_logprobs) / len(token_logprobs) / torch.log(
                    torch.tensor(2.0)
                ).item()
            )
        return metadata

    def query(self, query: str, num_responses: int = 1) -> List[Dict[str, Any]]:
        """
        每个 response 单独生成一次，便于统一“一个 thought/response 一次调用”的实验口径。
        """
        if self.cache and query in self.response_cache:
            cached = self.response_cache[query]
            self._set_last_response_metadata([dict(item["metadata"]) for item in cached])
            return cached

        prompt = self._build_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_token_count = int(inputs["input_ids"].shape[-1])

        responses: List[Dict[str, Any]] = []
        metadata_list: List[Dict[str, Any]] = []
        for _ in range(num_responses):
            generation_kwargs: Dict[str, Any] = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.do_sample and self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
            if self.do_sample and 0 < self.top_p < 1.0:
                generation_kwargs["top_p"] = self.top_p
            if self.do_sample and self.top_k > 0:
                generation_kwargs["top_k"] = self.top_k

            start_time = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**generation_kwargs)
            latency = time.perf_counter() - start_time
            self._record_model_call(latency)

            sequence = output.sequences[0]
            generated_token_ids = sequence[prompt_token_count:]
            text = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            ).strip()
            completion_token_count = int(generated_token_ids.shape[-1])
            self.prompt_tokens += prompt_token_count
            self.completion_tokens += completion_token_count
            self._calculate_cost()

            metadata = self._entropy_metadata_from_scores(
                list(output.scores),
                generated_token_ids,
            )
            metadata["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }
            metadata["latency_seconds"] = latency

            item = {"generated_text": text, "metadata": metadata}
            responses.append(item)
            metadata_list.append(metadata)

        self._set_last_response_metadata(metadata_list)
        if self.cache:
            self.response_cache[query] = responses
        return responses

    def get_response_texts(self, query_responses: List[Dict[str, Any]]) -> List[str]:
        metadata = [dict(item.get("metadata", {})) for item in query_responses]
        self._set_last_response_metadata(metadata)
        return [item["generated_text"] for item in query_responses]
