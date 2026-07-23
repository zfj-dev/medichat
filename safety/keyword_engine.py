"""关键词规则引擎"""
import yaml
from pathlib import Path
from typing import Optional, Dict


class KeywordSafetyEngine:
    def __init__(self, rules_path: str):
        with open(rules_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.rules = sorted(config["rules"], key=lambda r: r["priority"])

    def check(self, text: str) -> Optional[Dict]:
        """检查文本，返回命中的规则或 None"""
        for rule in self.rules:
            # 关键词匹配
            kws = rule.get("keywords", [])
            if not any(kw in text for kw in kws):
                continue

            # 如果有修饰词要求，需同时匹配
            mods = rule.get("modifiers", [])
            if mods and not any(m in text for m in mods):
                continue

            # 区域 + 感受组合匹配（针对胸痛类）
            zones = rule.get("zones", [])
            feelings = rule.get("feelings", [])
            if zones and feelings:
                if not (any(z in text for z in zones) and any(f in text for f in feelings)):
                    continue

            return {
                "rule_id": rule["id"],
                "action": rule["action"],
                "template": rule.get("template", ""),
            }

        # 兜底：心脏/胸口 + 痛/疼 组合
        danger_zones = ["胸口", "心脏", "心口", "心前区", "前胸"]
        danger_feelings = ["痛", "疼", "剧烈", "难受", "压迫", "不适", "不舒服", "紧缩", "刺痛"]
        if any(z in text for z in danger_zones) and any(f in text for f in danger_feelings):
            return {"rule_id": "fallback_chest", "action": "emergency", "template": "chest_pain"}

        return None


if __name__ == "__main__":
    engine = KeywordSafetyEngine(r"E:\medichat\safety\keyword_rules.yaml")

    tests = [
        ("感冒了吃什么药", "safe"),
        ("我胸口剧烈疼痛，喘不过气", "emergency"),
        ("今天有点胸闷", "safe"),           # 无修饰词
        ("我真的不想活了", "emergency"),
        ("头痛怎么办", "safe"),
        ("我胸口有点不舒服", "emergency"),   # 区域+感受组合
        ("安眠药吃多了会怎样", "emergency"),
        ("突然半身动不了了", "emergency"),
    ]

    for text, expected in tests:
        result = engine.check(text)
        actual = result["action"] if result else "safe"
        status = "✓" if actual == expected else "✗"
        print(f"{status} [{expected:10}] {text[:40]}")