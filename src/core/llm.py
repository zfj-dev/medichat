"""
MediChat LLM 引擎 —— Qwen2-7B + LoRA adapter
支持普通生成和 SSE 流式输出
"""
import torch
from typing import List, Dict, AsyncGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

from src.core.config import settings

SYSTEM_PROMPT = (
    "你是MediChat医疗科普助手。你只做医学科普，不提供诊断、处方或个人医疗建议。"
    "回答应通俗易懂、准确可靠。回答末尾请附免责声明。"
)


class LLMEngine:
    def __init__(self, model_path: str, lora_path: str = None):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        if lora_path:
            try:
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print(f"  LoRA adapter 已加载: {lora_path}")
            except Exception as e:
                print(f"  LoRA 加载失败: {e}，使用基座模型")

        self.model.eval()

    def _build_prompt(self, message: str, docs: List[Dict]) -> str:
        """构造带知识上下文的 prompt"""
        ctx = ""
        if docs:
            parts = []
            for i, d in enumerate(docs):
                parts.append(
                    f"参考资料{i+1}（{d.get('title', '未知')}，来源：{d.get('source', '未知')}）：\n"
                    f"{d['content'][:500]}"
                )
            ctx = "\n\n".join(parts)

        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n"
            f"以下参考资料可以帮助你回答问题：\n{ctx}<|im_end|>\n"
            f"<|im_start|>user\n{message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(self, message: str, docs: List[Dict]) -> str:
        """普通生成"""
        prompt = self._build_prompt(message, docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=True,
            )

        reply = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        return reply.strip()

    async def generate_stream(
        self, message: str, docs: List[Dict]
    ) -> AsyncGenerator[str, None]:
        """SSE 流式生成"""
        prompt = self._build_prompt(message, docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=True,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token
