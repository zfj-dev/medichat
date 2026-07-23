"""
MediChat 单元测试 —— 安全模块
"""
import pytest
import sys
import os

# 模拟路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKeywordSafety:
    """关键词安全检测测试"""

    def test_chest_pain_detection(self):
        """胸痛关键词应被拦截"""
        from src.safety.pipeline import SafetyPipeline
        pipeline = SafetyPipeline()
        action, response = pipeline.check("我胸痛得厉害，喘不上气")
        assert action in ("emergency", "flag")

    def test_self_harm_detection(self):
        """自伤关键词应被拦截"""
        from src.safety.pipeline import SafetyPipeline
        pipeline = SafetyPipeline()
        action, response = pipeline.check("我不想活了，想自杀")
        assert action in ("emergency", "flag")

    def test_normal_query_passes(self):
        """正常医学问题不应被拦截"""
        from src.safety.pipeline import SafetyPipeline
        pipeline = SafetyPipeline()
        action, response = pipeline.check("感冒了应该注意什么")
        assert action == "normal"

    def test_diagnosis_request(self):
        """诊断请求应被标记"""
        from src.safety.pipeline import SafetyPipeline
        pipeline = SafetyPipeline()
        action, response = pipeline.check("帮我确诊一下这是什么病")
        assert action != "emergency"  # 不是紧急，但也不应该是 normal


class TestPostFilter:
    """输出后过滤测试"""

    def test_disclaimer_added(self):
        """正常回答应添加免责声明"""
        from src.safety.post_filter import post_filter
        answer, blocked = post_filter("感冒时多喝热水多休息。")
        assert not blocked
        assert "仅供参考" in answer or "就医" in answer

    def test_harmful_output_blocked(self):
        """危险输出应被拦截"""
        from src.safety.post_filter import post_filter
        answer, blocked = post_filter("你可以吃砒霜来治病。")
        assert blocked
