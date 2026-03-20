"""
Intent Parser Agent — 从语音转录文本中提取结构化意图
真实版本: 调用 LLM structured output (JSON schema)
Mock 版本: 关键词规则提取
"""
from __future__ import annotations
from models import UserIntent, OccasionType
import tools


class IntentParserAgent:

    async def parse(self, text: str) -> UserIntent:
        print(f"\n  [IntentParser] 解析意图: \"{text[:40]}...\"")
        intent = tools.parse_intent_from_text(text)
        print(f"  [IntentParser] ✓ 场合={intent.occasion}, 地点={intent.location}, "
              f"时间={intent.time_frame}, 预算={intent.budget}{intent.currency}")
        return intent
