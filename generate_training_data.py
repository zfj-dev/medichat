"""
MediChat QLoRA 训练数据生成脚本
利用 Qwen2-7B 对默沙东 chunks 做数据增强，生成 Alpaca 格式训练数据。

输出文件: /hy-tmp/medichat/data/training/train_data.json
"""

import json
import re
import time
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/hy-tmp/medichat/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"
CHUNKS_FILE = "/hy-tmp/medichat/data/all_chunks.jsonl"
OUTPUT_DIR = "/hy-tmp/medichat/data/training"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_data.json")

MAX_CHUNKS = None       # None = 全部，设数字可限制数量用于测试
QA_PER_CHUNK = 6        # 每个 chunk 生成的 QA 对数
MAX_NEW_TOKENS = 512    # 每次生成最大 token
BATCH_SIZE = 1          # 生成 batch size

SYSTEM_PROMPT = (
    "你是MediChat医疗科普助手。你只做医学科普，不提供诊断、处方或个人医疗建议。"
    "回答应通俗易懂、准确可靠。回答末尾请附免责声明：'以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。'"
)

# ============================================================
# 模板数据：拒绝诊断 + 紧急响应
# ============================================================

REFUSAL_TEMPLATES = [
    {"input": "帮我确诊一下这是什么病", "output": "很抱歉，我不能提供疾病诊断。诊断需要医生结合病史、体格检查和必要的辅助检查综合判断。"
     "建议您到医院就诊，由专业医生进行评估。\n\n以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
    {"input": "我最近头疼，给我开点药吧", "output": "很抱歉，我不能开具处方或推荐具体药物。头痛可能由多种原因引起，"
     "不同病因需要不同的治疗方案。建议您到医院就诊，让医生明确诊断后进行针对性治疗。\n\n"
     "以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
    {"input": "我是不是得了癌症", "output": "很抱歉，我不能判断您是否患有癌症。癌症的诊断需要依靠影像学检查、"
     "病理活检等多种手段，而且很多其他疾病也可能出现类似症状。请不要过度焦虑，建议您及时到医院做全面检查。\n\n"
     "以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
    {"input": "我该不该去医院", "output": "我无法给您具体的就医建议，但一般来说，如果出现以下情况建议及时就医："
     "症状持续不缓解、疼痛剧烈、出现新的严重症状、或影响到日常生活。您可以描述您的具体情况，我可以提供相关的科普知识供参考。\n\n"
     "以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
    {"input": "帮我看一下这个检查报告", "output": "很抱歉，我无法解读医学检查报告。检查报告的解读需要专业医生结合患者的具体情况综合判断，"
     "不同指标的意义因人而异。建议您将报告带给就诊医生进行专业解读。\n\n"
     "以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
    {"input": "我家人有这个病，我是不是也会得", "output": "某些疾病确实有遗传倾向，但这并不意味着您一定会患病。遗传因素只是风险之一，"
     "生活方式、环境因素等同样重要。建议您咨询医生了解具体疾病的遗传风险，必要时可进行相关筛查。\n\n"
     "以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
    {"input": "给我推荐个最好的药", "output": "很抱歉，我不能推荐所谓的「最好」的药物。药物治疗需要个体化，同样的疾病在不同人身上可能需要不同的药物，"
     "还要考虑副作用、禁忌症、药物相互作用等因素。请咨询专业医生，根据您的具体情况选择合适的治疗方案。\n\n"
     "以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"},
]

EMERGENCY_TEMPLATES = [
    {"input": "我突然胸痛得厉害", "output": "您描述的胸痛可能是心脏问题等严重疾病的征兆，"
     "需要立即就医！请马上拨打120急救电话或前往最近的急诊科。在等待救援期间，请保持安静、"
     "不要剧烈活动，如果之前医生开过硝酸甘油等急救药物可按医嘱服用。\n\n"
     "⚠️ 此为紧急情况，请立即就医。以上内容仅供参考，不能替代专业医疗急救。"},
    {"input": "我呼吸困难喘不上气", "output": "呼吸困难可能由多种严重原因引起（如哮喘急性发作、过敏反应、心脏问题等），"
     "需要立即就医！请马上拨打120或前往最近的急诊科。如果身边有哮喘吸入剂等急救药物，可按医嘱使用。\n\n"
     "⚠️ 此为紧急情况，请立即就医。以上内容仅供参考，不能替代专业医疗急救。"},
    {"input": "我割伤了流了很多血止不住", "output": "大出血是紧急情况！请立即采取以下措施：用干净的布或绷带直接压迫伤口止血，"
     "将受伤部位抬高到心脏以上，并马上拨打120或前往最近的急诊科。不要自行拔除嵌入伤口的异物。\n\n"
     "⚠️ 此为紧急情况，请立即就医。以上内容仅供参考，不能替代专业医疗急救。"},
    {"input": "我吃了好多安眠药", "output": "药物过量是极度紧急的情况，可能危及生命！请立即拨打120急救电话，"
     "同时尽量保持清醒。如果身边有人，请让他们协助。告诉急救人员您服用的药物名称和大致数量。\n\n"
     "⚠️ 此为紧急情况，请立即就医。以上内容仅供参考，不能替代专业医疗急救。"},
]


def format_qa_pair(input_text: str, output_text: str) -> dict:
    """格式化为 LLaMA-Factory Alpaca 格式"""
    return {
        "instruction": SYSTEM_PROMPT,
        "input": input_text,
        "output": output_text,
    }


def generate_qa_pairs(chunk_text: str, chunk_title: str, model, tokenizer) -> list:
    """用 Qwen2-7B 基于一个 chunk 生成多个 QA 对"""
    prompt = f"""你是一个医疗数据标注专家。基于以下医学知识片段，生成 {QA_PER_CHUNK} 个不同角度的中文医疗科普问答对。

知识片段（来源：{chunk_title}）：
{chunk_text[:1500]}

要求：
1. 问题从不同角度提问（病因、症状、治疗、预防、注意事项等）
2. 回答基于知识片段内容，通俗易懂
3. 每个回答末尾加上免责声明："以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"
4. 格式：每个 QA 对以 "Q: " 开头的问题和 "A: " 开头的回答，QA 对之间用 "---" 分隔

请直接输出 QA 对，不要其他文字："""

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return parse_qa_pairs(response, chunk_title)


def parse_qa_pairs(response: str, source_title: str) -> list:
    """解析模型输出的 QA 对"""
    pairs = []
    blocks = response.split("---")

    for block in blocks:
        block = block.strip()
        q_match = re.search(r"Q[：:]\s*(.+?)\n", block)
        a_match = re.search(r"A[：:]\s*(.+)", block, re.DOTALL)

        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()

            # 确保有免责声明
            if "以上内容仅供参考" not in answer:
                answer += "\n\n以上内容仅供参考，不能替代专业医疗诊断。如有健康问题请及时就医。"

            pairs.append(format_qa_pair(question, answer))

    return pairs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  MediChat 训练数据生成")
    print(f"  模型: {MODEL_PATH}")
    print(f"  数据: {CHUNKS_FILE}")
    print(f"  每个 chunk 生成 {QA_PER_CHUNK} 个 QA 对")
    print("=" * 60)

    # 加载模型
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  模型加载完成, 设备: {model.device}")

    # 加载 chunks
    print("\n[2/4] 加载知识库...")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))

    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    print(f"  加载 {len(chunks)} 个文档片段")

    # 生成 QA 对
    print(f"\n[3/4] 生成训练数据 (预计 {len(chunks) * 5 // 60} 分钟)...")
    all_data = []

    for i, chunk in enumerate(chunks):
        title = chunk.get("title", "未知")
        content = chunk.get("content", "")
        if len(content) < 50:
            continue

        try:
            pairs = generate_qa_pairs(content, title, model, tokenizer)
            all_data.extend(pairs)

            if (i + 1) % 20 == 0 or i == len(chunks) - 1:
                print(f"  进度: {i+1}/{len(chunks)} | 已生成 {len(all_data)} 条")
        except Exception as e:
            print(f"  ⚠ chunk {i} ({title}) 生成失败: {e}")
            continue

        time.sleep(0.5)  # 避免显存波动

    # 添加模板数据
    print(f"\n  生成数据: {len(all_data)} 条")
    print("  添加模板数据...")
    for t in REFUSAL_TEMPLATES:
        all_data.append(format_qa_pair(t["input"], t["output"]))
    for t in EMERGENCY_TEMPLATES:
        all_data.append(format_qa_pair(t["input"], t["output"]))
    print(f"  总计: {len(all_data)} 条")

    # 保存
    print(f"\n[4/4] 保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  完成！共生成 {len(all_data)} 条训练数据")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
