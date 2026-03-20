"""
Wardrobe Analyzer Agent — 调用 AntiHallucinationPipeline 识别衣物
"""
from __future__ import annotations
from models import PerceptionResult
from pipeline import AntiHallucinationPipeline


class WardrobeAnalyzerAgent:

    def __init__(self):
        self.pipeline = AntiHallucinationPipeline(confidence_threshold=0.75)

    async def analyze(self, image: bytes) -> PerceptionResult:
        print("\n  [WardrobeAnalyzer] 启动衣物识别 Pipeline...")
        result = await self.pipeline.run(image)
        print(f"  [WardrobeAnalyzer] ✓ 最终识别: {len(result.confident_items)} 件衣物可用")
        return result
