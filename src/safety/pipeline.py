"""安全流水线 —— 串联关键词 → BERT → 后过滤"""
import sys
sys.path.insert(0, "/hy-tmp/medichat")

from safety.keyword_engine import check_keywords
from safety.classifier.inference import load as load_classifier, predict as classify
from safety.post_filter import filter as post_filter

_classifier_loaded = False

class SafetyPipeline:
    def __init__(self):
        global _classifier_loaded
        if not _classifier_loaded:
            try:
                load_classifier()
                _classifier_loaded = True
            except Exception:
                pass

    def check(self, text: str) -> tuple:
        """返回 (action, response)
        action: "normal" | "emergency" | "flag"
        """
        # 第一层：关键词
        action, resp = check_keywords(text)
        if action == "emergency":
            return action, resp

        # 第二层：BERT
        try:
            result = classify(text)
            if result["label"] == "urgent":
                return "emergency", "检测到可能的紧急情况。建议您立即就医。如有严重不适，请拨打120。"
            if result["label"] == "manipulative":
                return "flag", ""
        except Exception:
            pass

        return "normal", ""
