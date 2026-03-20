"""
Stylist Agent — 根据场合 + 天气 + 衣物列表推荐穿搭
真实版本: 多图输入 VLM，评估搭配协调度
Mock 版本: 规则匹配 + 工具函数
"""
from __future__ import annotations
from models import ClothingItem, OutfitRecommendation, UserIntent, WeatherInfo
import tools


class StylistAgent:

    async def recommend(
        self,
        clothing_items: list[ClothingItem],
        intent: UserIntent,
        weather: WeatherInfo,
    ) -> list[OutfitRecommendation]:
        print(f"\n  [Stylist] 为 '{intent.occasion}' 场合筛选穿搭方案...")
        print(f"  [Stylist] 天气: {weather.condition}, {weather.temperature_min}~{weather.temperature_max}°C")

        recommendations = tools.evaluate_outfit_for_occasion(
            clothing_items, intent, weather
        )

        if recommendations:
            best = recommendations[0]
            item_names = " + ".join(f"{i.color}{i.category}" for i in best.items)
            print(f"  [Stylist] ✓ 推荐方案 (匹配度 {best.match_score:.0%}): {item_names}")
        else:
            print("  [Stylist] ✗ 未找到合适穿搭，建议转入购物搜索")

        return recommendations
