"""
语言模型工厂模块：根据模型名称自动选择并实例化合适的语言模型类。
"""

from __future__ import annotations

from typing import Optional


def build_language_model(
        config_path: str,
        model_name: str,
        cache: bool = True,
        *,
        gemini_parallel_group_0based: Optional[int] = None,
):
    """
    根据模型名自动匹配并实例化语言模型类。

    命名约定：
    - `*-gcli`：使用 GCLI 代理（GCLIGemini），模型名去掉 `-gcli` 后缀
      - 例如：`gemini-2.5-flash-gcli` -> GCLIGemini(model_name='gemini-2.5-flash-gcli')
    - `gemini-*`：使用原生 Gemini SDK（Gemini）
    - `deepseek-*`：使用 DeepSeek API（DeepSeek）
    - 其它：默认使用 OpenAI/ChatGPT 兼容接口（ChatGPT）

    :param config_path: 配置文件路径
    :type config_path: str
    :param model_name: 模型名称
    :type model_name: str
    :param cache: 是否缓存响应，默认为 True
    :type cache: bool
    :param gemini_parallel_group_0based: 非 None 时原生 Gemini 使用多组 api_key（config 中 …-1/…-2），优先该组并在失败时切换
    :type gemini_parallel_group_0based: Optional[int]
    :return: 语言模型实例
    :rtype: AbstractLanguageModel
    """
    normalized = (model_name or "").strip()
    lower = normalized.lower()

    if lower.endswith("-gcli"):
        # 延迟导入，避免未安装或未使用时影响其它模型。
        from graph_of_thoughts.language_models.gcli_gemini import GCLIGemini
        return GCLIGemini(
            config_path,
            model_name=normalized,
            cache=cache,
        )
    
    if lower.startswith("gemini-"):
        if gemini_parallel_group_0based is not None:
            raise ImportError(
                "gemini_parallel_group_0based was requested, but "
                "gemini_grouped_failover.py is not present in this repository."
            )
        from graph_of_thoughts.language_models.gemini import Gemini

        return Gemini(
            config_path,
            model_name=normalized,
            cache=cache,
        )
    
    if lower.startswith("deepseek-"):
        from graph_of_thoughts.language_models.deepseek import DeepSeek

        return DeepSeek(
            config_path,
            model_name=normalized,
            cache=cache,
        )

    from graph_of_thoughts.language_models.chatgpt import ChatGPT

    return ChatGPT(
        config_path,
        model_name=normalized,
        cache=cache,
    )
