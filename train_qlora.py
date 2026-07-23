"""
MediChat QLoRA → FP16 LoRA 微调脚本
绕过 transformers 4.44.2 的 bitsandbytes bug，用 FP16 训练
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/hy-tmp/medichat/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"
DATA_FILE = "/hy-tmp/medichat/data/training/train_data.json"
OUTPUT_DIR = "/hy-tmp/medichat/models/lora-adapter"

BATCH_SIZE = 1            # FP16 需要更保守
GRAD_ACCUM = 16           # 有效 batch = 1 × 16 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 1024
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = (
    "你是MediChat医疗科普助手。你只做医学科普，不提供诊断、处方或个人医疗建议。"
    "回答应通俗易懂、准确可靠。回答末尾请附免责声明。"
)


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    texts = []
    for item in raw:
        instruction = item.get("instruction", SYSTEM_PROMPT)
        user_input = item.get("input", "")
        output = item.get("output", "")

        text = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


def tokenize(examples, tokenizer):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    print("=" * 60)
    print("  MediChat LoRA 微调 (FP16)")
    print(f"  模型: {MODEL_PATH}")
    print(f"  数据: {DATA_FILE}")
    print(f"  输出: {OUTPUT_DIR}")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    dataset = load_data()
    print(f"  训练样本: {len(dataset)} 条")

    # Tokenizer
    print("\n[2/4] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 模型 (FP16)
    print("\n[3/4] 加载模型 (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(f"  模型加载完成")

    # LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 分词
    print("\n  分词处理...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # 训练
    print("\n[4/4] 开始训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        ),
    )

    trainer.train()

    # 保存
    print(f"\n  保存 adapter 到 {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n{'='*60}")
    print(f"  训练完成！LoRA adapter: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
