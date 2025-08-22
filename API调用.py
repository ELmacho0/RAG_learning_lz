# -*- coding: utf-8 -*-
"""
multi_llm_basic.py
统一封装 Kimi（Moonshot）、豆包（Volc Ark）、通义千问（DashScope OpenAI 兼容）、DeepSeek 的基础对话调用。
默认以“逐字”方式流式输出；如需禁用流式输出，传 stream=False。

环境：Python 3.8+，仅依赖标准库 requests（pip install requests）
"""

import os
import sys
import json
import time
import typing as T
import requests

# ========== 1) 在此处填写/或用环境变量设置 各大模型的 API Key & Base URL ==========

# Kimi / Moonshot
MOONSHOT_API_KEY = os.getenv("sk-tN0ZLxKYNADE0TbIlmMfWMYR4mIdoyx92GGbQ5SkiOoMntB9", "sk-tN0ZLxKYNADE0TbIlmMfWMYR4mIdoyx92GGbQ5SkiOoMntB9")
# Moonshot 有全球与中国大陆两个域名，二选一：
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")  # 中国大陆
# MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"  # 全球

# 豆包 / 火山方舟 Ark（OpenAI 兼容）
DOUBAO_API_KEY = os.getenv("0044a172-02c1-443b-91a7-1c325dcd8e1c", "0044a172-02c1-443b-91a7-1c325dcd8e1c")
DOUBAO_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

# 通义千问 / 阿里云 Model Studio（DashScope OpenAI 兼容）
QWEN_API_KEY = os.getenv("sk-40119c25781c4cf7b12d778f68a3ae37", "sk-40119c25781c4cf7b12d778f68a3ae37")
# 按区域选择 Base URL（官方建议仅改 base_url、api_key、model 即可）
# 新加坡：
QWEN_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
# 中国北京可改为：
# QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek（OpenAI 兼容）
DEEPSEEK_API_KEY = os.getenv("sk-7061426ec317496fbbddc4cc368a931c", "sk-7061426ec317496fbbddc4cc368a931c")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ========== 2) 通用工具：逐字输出 & SSE 解析 ==========

def _print_char_by_char(text: str) -> None:
    """将文本逐字打印（不换行），尽量贴合“一个字一个字返回”的交互体验。"""
    for ch in text:
        print(ch, end="", flush=True)
        # 可按需加一点点延迟（体验更像“打字机”），例如：
        # time.sleep(0.001)
    # 不追加换行；由上层在结束时决定是否换行

def _iter_sse_lines(resp: requests.Response):
    # 统一按 UTF-8 解码 SSE 行，避免 requests 误判编码
    for raw in resp.iter_lines(decode_unicode=False):  # 注意 False
        if not raw:
            continue
        if raw.startswith(b"data:"):
            data = raw[len(b"data:"):].strip()
            if data == b"[DONE]":
                break
            yield data.decode("utf-8", errors="replace")  # 强制 UTF-8

def _finalize_stream(end: str = "\n"):
    """统一在流式结束时输出一个换行。"""
    if end:
        print(end, end="")

# ========== 3) 各厂商封装 ==========

def call_kimi_chat(
    model: str,
    messages: T.List[dict],
    temperature: float = 0.3,
    top_p: float = 1.0,
    max_tokens: T.Optional[int] = None,
    stop: T.Optional[T.Union[str, T.List[str]]] = None,
    stream: bool = True,
    timeout: int = 600,
    extra: T.Optional[dict] = None,
) -> str:
    """
    Kimi / Moonshot Chat Completions 基础调用（OpenAI 风格）
    Endpoint: {MOONSHOT_BASE_URL}/chat/completions
    认证：Authorization: Bearer <MOONSHOT_API_KEY>

    支持参数（参考官方 & 第三方权威文档）：
    - model: str
        例如 "moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2-0711-preview" 等。
    - messages: List[dict]
        OpenAI 消息格式 [{"role":"user","content":"..."}]。
    - temperature: float ∈ [0, 1]   ← 注意 Moonshot 限定[0,1]（官方说明）
    - top_p: float ∈ (0, 1]
    - max_tokens: int > 0（可选）
    - stop: str | List[str]（最多 16 条，SSE 或非流式均可）
    - stream: bool（默认 True，使用 SSE 流式）
    - extra: dict（透传给请求体的其他键）
    * 工具调用等高级能力此处未覆盖，仅做基础对话演示。

    返回：若 stream=True，函数边收边“逐字”打印，最终返回完整文本；
         若 stream=False，直接返回完整文本并一次性打印。
    """
    # Clamp/校验
    temperature = max(0.0, min(1.0, temperature))  # Moonshot 特别限制
    if top_p is not None:
        top_p = max(0.0, min(1.0, top_p))

    url = f"{MOONSHOT_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {MOONSHOT_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if stop is not None:
        payload["stop"] = stop
    if extra:
        payload.update(extra)

    out = []
    with requests.post(url, headers=headers, data=json.dumps(payload), stream=stream, timeout=timeout) as r:
        r.raise_for_status()
        if stream:
            for data in _iter_sse_lines(r):
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        _print_char_by_char(piece)
                        out.append(piece)
                except Exception:
                    # 不中断流；忽略异常行
                    continue
            _finalize_stream()
        else:
            obj = r.json()
            text = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            _print_char_by_char(text)
            _finalize_stream()
            out.append(text)
    return "".join(out)


def call_doubao_chat(
    model: str,
    messages: T.List[dict],
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: T.Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop: T.Optional[T.Union[str, T.List[str]]] = None,
    stream: bool = True,
    timeout: int = 600,
    extra: T.Optional[dict] = None,
) -> str:
    """
    豆包 / 火山方舟 Ark Chat Completions（OpenAI 兼容）
    Endpoint: {DOUBAO_BASE_URL}/chat/completions
    认证：Authorization: Bearer <ARK_API_KEY>

    支持参数（Ark 明确标注“OpenAI 兼容”，故沿用标准语义）：
    - model: str
        例如 "ep-xxxxxxxx"（推理接入点模型 ID）或官方直连模型名（若开通）。
    - messages: List[dict]
    - temperature: float ∈ [0, 2]（OpenAI 兼容惯例）
    - top_p: float ∈ (0, 1]
    - max_tokens: int > 0（可选）
    - presence_penalty: float ∈ [-2, 2]
    - frequency_penalty: float ∈ [-2, 2]
    - stop: str | List[str]
    - stream: bool（默认 True，SSE 流式）
    - extra: dict 透传

    说明：
    - Ark v3 提供 OpenAI 兼容的 /chat/completions，常见 Base URL：
      https://ark.cn-beijing.volces.com/api/v3
    - 若调用“豆包智能体（bots）”，路径不同（/api/v3/bots/chat/completions），本文不涉及。

    返回：与其他厂商封装一致。
    """
    # Clamp
    if top_p is not None:
        top_p = max(0.0, min(1.0, top_p))
    presence_penalty = max(-2.0, min(2.0, presence_penalty))
    frequency_penalty = max(-2.0, min(2.0, frequency_penalty))

    url = f"{DOUBAO_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if stop is not None:
        payload["stop"] = stop
    if extra:
        payload.update(extra)

    out = []
    with requests.post(url, headers=headers, data=json.dumps(payload), stream=stream, timeout=timeout) as r:
        r.raise_for_status()
        if stream:
            for data in _iter_sse_lines(r):
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        _print_char_by_char(piece)
                        out.append(piece)
                except Exception:
                    continue
            _finalize_stream()
        else:
            obj = r.json()
            text = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            _print_char_by_char(text)
            _finalize_stream()
            out.append(text)
    return "".join(out)


def call_qwen_chat(
    model: str,
    messages: T.List[dict],
    temperature: float = 0.8,
    top_p: float = 1.0,
    max_tokens: T.Optional[int] = None,
    stop: T.Optional[T.Union[str, T.List[str]]] = None,
    stream: bool = True,
    timeout: int = 600,
    extra: T.Optional[dict] = None,
) -> str:
    """
    通义千问（阿里云 Model Studio / DashScope）OpenAI 兼容 Chat Completions
    Endpoint（OpenAI 兼容）：
        新加坡: {QWEN_BASE_URL}/chat/completions
        北京:   https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    认证：Authorization: Bearer <DASHSCOPE_API_KEY>

    支持参数（OpenAI 兼容）：
    - model: str（如 "qwen-plus", "qwen-turbo", "qwen-max" 等）
    - messages: List[dict]
    - temperature: float ∈ [0, 2]
    - top_p: float ∈ (0, 1]
    - max_tokens: int > 0（可选）
    - stop: str | List[str]
    - stream: bool（默认 True，SSE 流式）
    - extra: dict 透传（如 stream_options={"include_usage": True}）

    返回：同上。
    """
    if top_p is not None:
        top_p = max(0.0, min(1.0, top_p))

    url = f"{QWEN_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if stop is not None:
        payload["stop"] = stop
    if extra:
        payload.update(extra)

    out = []
    with requests.post(url, headers=headers, data=json.dumps(payload), stream=stream, timeout=timeout) as r:
        r.raise_for_status()
        if stream:
            for data in _iter_sse_lines(r):
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        _print_char_by_char(piece)
                        out.append(piece)
                except Exception:
                    continue
            _finalize_stream()
        else:
            obj = r.json()
            text = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            _print_char_by_char(text)
            _finalize_stream()
            out.append(text)
    return "".join(out)


def call_deepseek_chat(
    model: str,
    messages: T.List[dict],
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: T.Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop: T.Optional[T.Union[str, T.List[str]]] = None,
    stream: bool = True,
    timeout: int = 600,
    extra: T.Optional[dict] = None,
) -> str:
    """
    DeepSeek Chat Completions（OpenAI 兼容）
    Endpoint: {DEEPSEEK_BASE_URL}/chat/completions
    认证：Authorization: Bearer <DEEPSEEK_API_KEY>

    支持参数（官方文档明确范围）：
    - model: str，例如：
        "deepseek-chat"（对应 DeepSeek-V3）
        "deepseek-reasoner"（对应 DeepSeek-R1）
    - messages: List[dict]
    - temperature: float ∈ [0, 2]（默认 1）
    - top_p: float ∈ (0, 1]（默认 1）
    - max_tokens: int ∈ [1, 8192]（不填默认为 4096）
    - presence_penalty: float ∈ [-2, 2]
    - frequency_penalty: float ∈ [-2, 2]
    - stop: str | List[str]（最多 16）
    - stream: bool（默认 True，SSE 流式；末尾含 data:[DONE]）
    - extra: dict（如 response_format={"type": "json_object"}、tools、tool_choice 等）

    返回：同上。
    """
    if top_p is not None:
        top_p = max(0.0, min(1.0, top_p))
    presence_penalty = max(-2.0, min(2.0, presence_penalty))
    frequency_penalty = max(-2.0, min(2.0, frequency_penalty))

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if stop is not None:
        payload["stop"] = stop
    if extra:
        payload.update(extra)

    out = []
    with requests.post(url, headers=headers, data=json.dumps(payload), stream=stream, timeout=timeout) as r:
        r.raise_for_status()
        if stream:
            for data in _iter_sse_lines(r):
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        _print_char_by_char(piece)
                        out.append(piece)
                except Exception:
                    continue
            _finalize_stream()
        else:
            obj = r.json()
            text = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            _print_char_by_char(text)
            _finalize_stream()
            out.append(text)
    return "".join(out)


# ========== 4) 统一调度函数（按 provider 路由） ==========

def run_chat(
    provider: str,
    model: str,
    prompt: T.Union[str, T.List[dict]],
    # 常用参数（其余通过 extra 透传）
    temperature: T.Optional[float] = None,
    top_p: T.Optional[float] = None,
    max_tokens: T.Optional[int] = None,
    stream: bool = True,
    stop: T.Optional[T.Union[str, T.List[str]]] = None,
    presence_penalty: T.Optional[float] = None,
    frequency_penalty: T.Optional[float] = None,
    extra: T.Optional[dict] = None,
) -> str:
    """
    统一入口：
    - provider: "kimi" | "豆包" | "千问" | "deepseek"
    - model: 厂商具体模型名
    - prompt: str（将自动包装为 [{"role":"user","content": prompt}]）
              或 OpenAI 风格的 messages: List[dict]
    - 其他参数：仅填与当前厂商相符的那些；不需要的留空。
    返回：完整文本（即便流式打印了，也会汇总返回）
    """
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    provider = provider.lower()
    kw = dict(
        model=model,
        messages=messages,
        stream=stream,
        extra=extra or {},
    )

    if temperature is not None:
        kw["temperature"] = temperature
    if top_p is not None:
        kw["top_p"] = top_p
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if stop is not None:
        kw["stop"] = stop
    if presence_penalty is not None:
        kw["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kw["frequency_penalty"] = frequency_penalty

    if provider == "kimi":
        return call_kimi_chat(**kw)
    elif provider == "豆包":
        return call_doubao_chat(**kw)
    elif provider in ("千问", "通义千问"):
        return call_qwen_chat(**kw)
    elif provider == "deepseek":
        return call_deepseek_chat(**kw)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use one of: kimi | 豆包 | 千问 | deepseek")


# ========== 5) 示例：在 __main__ 中通过“参数方式”调用 ==========

if __name__ == "__main__":
    """
    运行示例：
    1) 先在本文件顶部填好各家的 API Key（或用环境变量）
    2) 在下方选择你要试的 provider / model / 参数 / 提示词
    3) python multi_llm_basic.py
    """

    # ——示例 A：DeepSeek（流式，逐字打印）——
    run_chat(
        provider="豆包",
        model="doubao-1-5-thinking-pro-m-250428",           # 或 "deepseek-reasoner"
        prompt="帮我写一个可以调用各大模型API的Python，需要包括KIMI、豆包、通义千问、DeepSeek 在开头的位置填写各大模型的aipkey，在每个模型接口内部写清楚都支持那些参数，以及参数的选择范围。 在__name__ = main 后边写调用函数，模型、模型的参数和提示词都通过参数的方法调用，仅支持基本功能即可，全部默认一个字一个字返回。",
        temperature=0.1,
        top_p=1.0,
        max_tokens=3000,
        stream=True,                     # 默认 True：逐字输出
    )

    # ——示例 B：通义千问（一次性返回，可改为 stream=True）——
    # run_chat(
    #     provider="qwen",
    #     model="qwen-plus",
    #     prompt="给我一份 5 条的 Python 代码优化建议清单。",
    #     temperature=0.6,
    #     max_tokens=400,
    #     stream=False,  # 关闭流式，一次性打印
    # )

    # ——示例 C：豆包 Ark（ep-xxxx 推理接入点）——
    # run_chat(
    #     provider="doubao",
    #     model="ep-20250101000000-xxxx",   # 替换成你的推理接入点模型 ID
    #     prompt="请用要点列出敏捷开发迭代的关键实践。",
    #     temperature=1.0,
    #     presence_penalty=0.0,
    #     frequency_penalty=0.0,
    #     stream=True,
    # )

    # ——示例 D：Kimi / Moonshot（注意 temperature ∈ [0,1]）——
    # run_chat(
    #     provider="kimi",
    #     model="moonshot-v1-8k",  # 或 kimi-k2-0711-preview 等
    #     prompt="把下列句子改写得更专业简洁：我已经完成了报告的初稿。",
    #     temperature=0.3,         # Kimi 仅 [0,1]
    #     top_p=0.95,
    #     stream=True,
    # )
