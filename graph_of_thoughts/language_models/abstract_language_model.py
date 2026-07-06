# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import json
import os
import logging
import math
import time


class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False
    ) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the language model. Defaults to "".
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
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
        self.last_response_metadata: List[Dict[str, Any]] = []

    def _now(self) -> float:
        """
        Return the current wall-clock time for latency measurements.
        """
        return time.perf_counter()

    def _calculate_cost(self) -> None:
        """
        Recalculate cumulative cost from cumulative token counters when the
        concrete model exposes OpenAI-style token prices.
        """
        prompt_token_cost = getattr(self, "prompt_token_cost", 0.0)
        response_token_cost = getattr(self, "response_token_cost", 0.0)
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            prompt_token_cost * prompt_tokens_k
            + response_token_cost * completion_tokens_k
        )

    def _entropy_from_top_logprobs(self, top_logprobs: List[Any]) -> Union[float, None]:
        """
        Approximate token entropy from a provider's top-logprobs list. This is a
        lower-bound style estimate if the provider returns only top-k tokens.
        """
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

    def _extract_choice_logprob_metadata(self, choice: Any) -> Dict[str, Any]:
        """
        Extract token-level observability fields from OpenAI-compatible choices.
        The exact logprobs schema differs slightly across providers, so this
        method uses defensive attribute/dict access.
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

        metadata: Dict[str, Any] = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "num_observed_tokens": len(content_logprobs),
        }
        if token_logprobs:
            metadata["avg_neg_logprob_bits"] = (
                -sum(token_logprobs) / len(token_logprobs) / math.log(2)
            )
        if token_entropies:
            metadata["token_entropies_bits"] = token_entropies
            metadata["sum_entropy_bits"] = sum(token_entropies)
            metadata["avg_entropy_bits"] = sum(token_entropies) / len(token_entropies)
            metadata["entropy_estimate"] = "top_logprobs"
        return metadata

    def _set_last_response_metadata(
        self,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Store metadata aligned with the list returned by get_response_texts().
        """
        self.last_response_metadata = metadata

    def consume_last_response_metadata(
        self, expected_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Return metadata from the most recent get_response_texts() call and pad or
        trim it to the number of response texts the caller is handling.
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
        Build per-choice metadata for OpenAI-compatible ChatCompletion objects.
        """
        if not isinstance(query_response, list):
            query_response = [query_response]

        metadata: List[Dict[str, Any]] = []
        for response in query_response:
            usage = getattr(response, "usage", None)
            latency = getattr(response, "_got_latency_seconds", None)
            response_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            for choice in getattr(response, "choices", []) or []:
                choice_metadata: Dict[str, Any] = {
                    "provider": provider,
                    "model": getattr(self, "model_id", self.model_name),
                    "usage": response_usage,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "has_logprobs": getattr(choice, "logprobs", None) is not None,
                    "latency_seconds": latency,
                }
                choice_metadata.update(self._extract_choice_logprob_metadata(choice))
                metadata.append(choice_metadata)
        return metadata

    def load_config(self, path: str) -> None:
        """
        Load configuration from a specified path.

        :param path: Path to the config file. If an empty path provided,
                     default is `config.json` in the current directory.
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
        Clear the response cache.
        """
        self.response_cache.clear()

    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        Abstract method to extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[Any], Any]
        :return: List of textual responses.
        :rtype: List[str]
        """
        pass
