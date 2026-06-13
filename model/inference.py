"""批量推理接口：vLLM 批量生成和 logprobs 获取。"""

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 0.95
    stop_tokens: List[str] = field(
        default_factory=lambda: ["</tool_call>", "</answer>"]
    )
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    text: str
    logprobs: Optional[List[float]] = None
    tokens: Optional[List[int]] = None
    finish_reason: str = ""


class BatchInference:
    """vLLM 批量推理接口。

    RTX 4090 性能: ~1500 tok/s, 单条推理 ~0.5s.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_model_len: int = 2048,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        """Lazy-init vLLM 引擎。"""
        if self._engine is not None:
            return self._engine

        from vllm import LLM

        self._engine = LLM(
            model=self.model_name_or_path,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        logger.info(f"vLLM engine ready: {self.model_name_or_path}")
        return self._engine

    def generate(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> List[str]:
        """批量生成文本。"""
        engine = self._ensure_engine()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        outputs = engine.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def generate_with_logprobs(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        """批量生成并返回每 token 的 logprob（GRPO 训练用）。"""
        engine = self._ensure_engine()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,
            **kwargs,
        )
        outputs = engine.generate(prompts, sampling_params)

        results: List[GenerationResult] = []
        for o in outputs:
            r = o.outputs[0]
            results.append(GenerationResult(
                text=r.text,
                logprobs=(
                    [t.logprob for t in r.logprobs]
                    if r.logprobs else None
                ),
                tokens=r.token_ids,
                finish_reason=r.finish_reason or "",
            ))
        return results

    async def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """流式生成（Demo 用）。"""
        from vllm import SamplingParams

        engine = self._ensure_engine()
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async for output in engine.generate(
            prompt, sampling_params, request_id="demo"
        ):
            yield output.outputs[0].text
