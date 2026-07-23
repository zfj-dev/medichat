"""安全流水线 —— 串联关键词 → BERT → 后过滤"""
import sys
sys.path.insert(0, "/hy-tmp/medichat")

from safety.keyword_engine import KeywordSafetyEngine
from safety.classifier.inference import load as load_classifier, predict as classify
from safety.post_filter import filter as post_filter
from pathlib import Path

_classifier_loaded = False
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        rules_path = Path("/hy-tmp/medichat/safety/keyword_rules.yaml")
        if rules_path.exists():
            _engine = KeywordSafetyEngine(str(rules_path))
        else:
            _engine = KeywordSafetyEngine("/hy-tmp/medichat/safety/keyword_rules.yaml")
    return _engine


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
        engine = _get_engine()

        # 第一层：关键词
        result = engine.check(text)
        if result and result.get("action") == "emergency":
            return "emergency", result.get("template", "检测到紧急情况，请立即就医。")

        # 第二层：BERT
        try:
            bert_result = classify(text)
            if bert_result.get("label") == "urgent":
                return "emergency", "检测到可能的紧急情况。建议您立即就医。如有严重不适，请拨打120。"
            if bert_result.get("label") == "manipulative":
                return "flag", ""
        except Exception:
            pass

        return "normal", ""
