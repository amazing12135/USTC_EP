"""模型加载器：QLoRA 训练模型和 vLLM 推理引擎的加载。

支持两种模式：
- 训练模式：Base + QLoRA 4-bit（用于 SFT / GRPO 策略更新）
- 推理模式：合并 LoRA → vLLM 引擎（用于高效 rollout 采样）

GRPO 双模型架构：
  policy_model (LoRA, trainable) ← 策略更新
  ref_model    (LoRA, frozen)    ← KL 约束参考
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器。"""

    DEFAULT_LORA_CONFIG: Dict[str, Any] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }

    DEFAULT_BNB_CONFIG: Dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else os.environ.get("HF_HOME")
        self._tokenizer: Any = None

    # ================================================================
    # 训练模式
    # ================================================================

    def load_for_training(
        self,
        lora_config: Optional[Dict[str, Any]] = None,
        bnb_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """加载 Base 模型 + QLoRA 4-bit + LoRA adapter。

        Returns:
            带 LoRA 的 PeftModel
        """
        lora_cfg = {**self.DEFAULT_LORA_CONFIG, **(lora_config or {})}
        bnb_cfg = {**self.DEFAULT_BNB_CONFIG, **(bnb_config or {})}

        logger.info(f"Loading {self.model_name} for training (QLoRA 4-bit)")
        self._check_gpu_memory()

        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        quant_config = BitsAndBytesConfig(
            load_in_4bit=bnb_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
        )

        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
            target_modules=lora_cfg["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info("Model loaded for training")
        return model

    def load_ref_model(
        self,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """加载参考模型（冻结 LoRA，用于 GRPO KL 约束）。"""
        model = self.load_for_training(lora_config=lora_config)
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Reference model loaded (frozen)")
        return model

    # ================================================================
    # 推理模式
    # ================================================================

    def load_for_inference(
        self,
        lora_path: Optional[str | Path] = None,
        max_model_len: int = 2048,
    ) -> Any:
        """加载模型到 vLLM 引擎。

        如果有 lora_path：自动 merge → 加载 merged model。
        如果无 lora_path：直接加载 base model。
        """
        logger.info(f"Loading {self.model_name} for vLLM inference")

        model_to_load = self.model_name

        if lora_path is not None:
            merged_path = Path(lora_path).parent / "merged_model"
            if not merged_path.exists():
                logger.info(f"Merging LoRA from {lora_path} -> {merged_path}")
                self.merge_and_save(lora_path, merged_path)
            model_to_load = str(merged_path)

        from vllm import LLM

        llm = LLM(
            model=model_to_load,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        logger.info("vLLM engine initialized")
        return llm

    # ================================================================
    # LoRA 合并
    # ================================================================

    def merge_and_save(
        self,
        lora_weights: str | Path,
        output_path: str | Path,
    ) -> None:
        """合并 LoRA adapter 到 Base 模型，以 safetensors 保存。

        合并后可直接被 vLLM 加载，无需 adapter 推理开销。
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Merging LoRA: {lora_weights} -> {output_path}")

        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        model = PeftModel.from_pretrained(base_model, str(lora_weights))
        model = model.merge_and_unload()

        model.save_pretrained(
            str(output_path),
            safe_serialization=True,
            max_shard_size="5GB",
        )
        self._save_tokenizer(output_path)

        logger.info(f"Merge complete: {output_path}")
        del model
        torch.cuda.empty_cache()

    # ================================================================
    # Tokenizer
    # ================================================================

    def load_tokenizer(self) -> Any:
        """加载与模型配套的 tokenizer。"""
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Tokenizer loaded")
        return self._tokenizer

    def _save_tokenizer(self, output_dir: Path) -> None:
        tokenizer = self.load_tokenizer()
        tokenizer.save_pretrained(str(output_dir))

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """获取 GPU 信息。"""
        info: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
        }
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["memory_total_gb"] = (
                torch.cuda.get_device_properties(0).total_mem / (1024**3)
            )
        return info

    @staticmethod
    def _check_gpu_memory() -> None:
        total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = total - allocated
        logger.info(f"GPU memory: {free:.1f} GB free / {total:.1f} GB total")
