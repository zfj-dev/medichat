"""输出后过滤器 —— 安全防御最后一层"""
import re

DISCLAIMER = "\n\n⚠️ 以上内容仅供参考，不能替代专业医疗诊断。如有健康问题，请咨询医生。"

DANGEROUS_PATTERNS = [
    (re.compile(r"你[确肯应].*得了"), "疑似诊断"),
    (re.compile(r"建议.*服用.*[药片胶囊颗粒]"), "药物建议"),
    (re.compile(r"肯定.*能.*治好|绝对.*有效|保证.*治愈"), "疗效保证"),
    (re.compile(r"你的[病证症].*是"), "确诊表述"),
    (re.compile(r"不需要.*看.*医生|不用.*去.*医院"), "劝阻就医"),
    (re.compile(r"这个药.*对.*你.*有效"), "个性化用药"),
    (re.compile(r"你的病.*应该"), "治疗建议"),
]


def has_disclaimer(text):
    return any(kw in text for kw in ["仅供参考", "不能替代", "请咨询医生"])


def filter(text):
    """过滤危险表述，追加免责声明"""
    for pattern, issue in DANGEROUS_PATTERNS:
        if pattern.search(text):
            return (
                f"抱歉，系统检测到回复可能包含不当内容（{issue}），已自动拦截。如需专业建议，请咨询医生。{DISCLAIMER}",
                True
            )

    if not has_disclaimer(text):
        text = text.rstrip() + DISCLAIMER

    return text, False