"""
外部工具接口定义 + Mock 实现

USE_MOCK=true  (默认) 使用 Mock，无需真实 API Key
USE_MOCK=false 使用真实 API（需配置相应 Key）
"""
from __future__ import annotations
import asyncio
import os
import random
import uuid
from typing import Any

from models import (
    ClothingItem, Detection, WeatherInfo, ShoppingResult, UserIntent, OccasionType
)

USE_MOCK = os.getenv("USE_MOCK", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Mock 数据库 — 模拟衣橱中的衣物
# ---------------------------------------------------------------------------
MOCK_WARDROBE: list[dict] = [
    {
        "category": "西装外套",
        "color": "深蓝色",
        "style": "正式",
        "material": "羊毛混纺",
        "condition": "good",
        "vlm_results": ["深蓝色西装外套", "深蓝色西装外套", "深蓝色正装外套"],
        "cv_category": "blazer",
        "det_conf": 0.92,
    },
    {
        "category": "白色衬衫",
        "color": "白色",
        "style": "正式",
        "material": "棉",
        "condition": "good",
        "vlm_results": ["白色衬衫", "白色正式衬衫", "白色衬衫"],
        "cv_category": "shirt",
        "det_conf": 0.88,
    },
    {
        "category": "西裤",
        "color": "深灰色",
        "style": "正式",
        "material": "涤纶",
        "condition": "good",
        "vlm_results": ["深灰色西裤", "灰色西裤", "深灰色正式裤"],
        "cv_category": "trousers",
        "det_conf": 0.85,
    },
    {
        "category": "牛仔裤",
        "color": "蓝色",
        "style": "休闲",
        "material": "棉",
        "condition": "good",
        "vlm_results": ["蓝色牛仔裤", "牛仔裤", "休闲牛仔裤"],
        "cv_category": "jeans",
        "det_conf": 0.90,
    },
    {
        "category": "运动外套",
        "color": "黑色",
        "style": "运动",
        "material": "聚酯纤维",
        "condition": "good",
        "vlm_results": ["黑色运动外套", "运动夹克", "黑色运动夹克"],
        "cv_category": "jacket",
        "det_conf": 0.78,   # 置信度较低，将触发交叉验证
    },
    {
        "category": "不明衣物",
        "color": "uncertain",
        "style": "uncertain",
        "material": "uncertain",
        "condition": "folded",
        "vlm_results": ["灰色毛衣", "灰色帽衫", "灰色外套"],   # VLM 意见不一
        "cv_category": "sweater",
        "det_conf": 0.61,   # 低置信，会进入 confirmation_cards
    },
]

MOCK_SHOPPING: list[dict] = [
    {
        "name": "Banana Republic 商务科技外套",
        "price": 178.0,
        "url": "https://example.com/br-tech-jacket",
        "image_url": "https://example.com/br-tech-jacket.jpg",
        "reason": "商务休闲风格，适合科技峰会，防水面料适应西雅图多雨气候",
        "rating": 4.5,
    },
    {
        "name": "Everlane 都市外套",
        "price": 148.0,
        "url": "https://example.com/everlane-city-jacket",
        "image_url": "https://example.com/everlane-city-jacket.jpg",
        "reason": "简约科技感设计，符合 AI 峰会场合，价格在预算内",
        "rating": 4.3,
    },
    {
        "name": "Uniqlo 混纺羊毛大衣",
        "price": 199.0,
        "url": "https://example.com/uniqlo-wool-coat",
        "image_url": "https://example.com/uniqlo-wool-coat.jpg",
        "reason": "专业正式感，西雅图春季保暖，Uniqlo 性价比高",
        "rating": 4.6,
    },
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

async def speech_to_text(audio: bytes) -> str:
    """ASR: 语音转文字 (Whisper)"""
    if USE_MOCK:
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return "我下周要去西雅图参加一个重要的AI科技峰会，帮我从衣橱里挑一套合适的。如果没有合适的，帮我在网上挑一件预算200美元以内的外套。"
    else:
        # 真实实现: import openai; client.audio.transcriptions.create(...)
        raise NotImplementedError("需要配置 OPENAI_API_KEY")


async def enhance_image(image: bytes) -> bytes:
    """图像预处理增强 (CLAHE + Real-ESRGAN)"""
    if USE_MOCK:
        await asyncio.sleep(0.05)
        return image  # Mock: 直接返回原图
    else:
        # 真实实现: cv2.createCLAHE(...).apply(gray)
        raise NotImplementedError("需要安装 opencv-python")


async def detect_clothing_regions(image: bytes) -> list[Detection]:
    """
    传统 CV 目标检测 — Grounding-DINO (mock)
    返回每件衣物的 bbox 和检测置信度
    """
    if USE_MOCK:
        await asyncio.sleep(0.2)
        detections = []
        for i, item in enumerate(MOCK_WARDROBE):
            detections.append(Detection(
                det_id=f"det_{i}",
                bbox=(0.1 * i % 0.8, 0.1, 0.15, 0.3),  # mock bbox
                detection_confidence=item["det_conf"],
            ))
        return detections
    else:
        raise NotImplementedError("需要安装 groundingdino")


async def segment_clothing(image: bytes, detections: list[Detection]) -> list[bytes]:
    """
    实例分割 — SAM2 (mock)
    返回每件衣物的裁剪图
    """
    if USE_MOCK:
        await asyncio.sleep(0.15)
        return [b"mock_crop_" + str(i).encode() for i in range(len(detections))]
    else:
        raise NotImplementedError("需要安装 segment_anything")


async def classify_clothing_vlm(crop: bytes, model_id: str = "gpt4o") -> dict:
    """
    单件衣物属性识别 — VLM (mock)
    模拟 3 个不同 VLM 各返回一份识别结果
    """
    if USE_MOCK:
        await asyncio.sleep(0.1)
        # 用 crop 索引对应 mock 数据
        idx = int(crop.decode().replace("mock_crop_", ""))
        item = MOCK_WARDROBE[idx]
        # 不同 VLM 返回略有差异的结果
        vlm_idx = {"gpt4o": 0, "claude": 1, "gemini": 2}.get(model_id, 0)
        raw = item["vlm_results"][vlm_idx % len(item["vlm_results"])]
        return {
            "raw_description": raw,
            "category": item["category"] if "uncertain" not in item["color"] else raw.split()[-1],
            "color": item["color"],
            "style": item["style"],
            "material": item["material"],
            "model": model_id,
        }
    else:
        raise NotImplementedError("需要配置 VLM API Key")


async def classify_clothing_cv(crop: bytes) -> dict:
    """
    传统 CV 分类器 — EfficientNet (mock)
    作为 VLM 的第二意见
    """
    if USE_MOCK:
        await asyncio.sleep(0.05)
        idx = int(crop.decode().replace("mock_crop_", ""))
        item = MOCK_WARDROBE[idx]
        return {
            "category": item["cv_category"],
            "confidence": item["det_conf"],
        }
    else:
        raise NotImplementedError("需要安装 torchvision + 微调模型权重")


async def get_weather_forecast(location: str, date_range: str) -> WeatherInfo:
    """天气预报 API — OpenWeatherMap (mock)"""
    if USE_MOCK:
        await asyncio.sleep(0.1)
        return WeatherInfo(
            location=location,
            date_range=date_range,
            temperature_min=8.0,
            temperature_max=14.0,
            condition="Partly Cloudy with Rain",
            precipitation_prob=0.65,
            wind_speed=20.0,
        )
    else:
        raise NotImplementedError("需要配置 OPENWEATHERMAP_API_KEY")


async def search_products(
    query: str, max_price: float, category: str = "jacket"
) -> list[ShoppingResult]:
    """电商搜索 — Google Shopping API (mock)"""
    if USE_MOCK:
        await asyncio.sleep(0.2)
        results = []
        for p in MOCK_SHOPPING:
            if p["price"] <= max_price:
                results.append(ShoppingResult(
                    product_name=p["name"],
                    price=p["price"],
                    currency="USD",
                    url=p["url"],
                    image_url=p["image_url"],
                    match_reason=p["reason"],
                    rating=p["rating"],
                ))
        return results
    else:
        raise NotImplementedError("需要配置 SERPAPI_KEY 或 Google Shopping API")


def parse_intent_from_text(text: str) -> UserIntent:
    """
    意图解析 (mock — 真实版本调用 LLM structured output)
    从文本提取结构化意图
    """
    # 真实版本: 调用 LLM 返回 JSON schema
    # 这里 mock 关键词匹配
    location = "西雅图" if "西雅图" in text else "未知"
    occasion = "AI科技峰会" if "峰会" in text or "科技" in text else "商务"
    budget = 200.0
    time_frame = "下周" if "下周" in text else "近期"
    fallback = "网购外套" if "网上" in text else "无"

    return UserIntent(
        occasion=occasion,
        location=location,
        time_frame=time_frame,
        budget=budget,
        currency="USD",
        fallback_action=fallback,
        occasion_type=OccasionType.TECH_CONFERENCE,
    )


def evaluate_outfit_for_occasion(
    items: list[ClothingItem], intent: UserIntent, weather: WeatherInfo
) -> list[Any]:
    """
    穿搭匹配评估 (mock — 真实版本调用 VLM 多图评估)
    根据场合 + 天气评分每件衣物的适合度
    """
    from models import OutfitRecommendation  # 避免循环导入

    outfit_candidates = []

    # 策略: 优先找 "正式" 风格的套装
    formal_tops = [i for i in items if i.style == "正式" and i.category in ("西装外套", "白色衬衫")]
    formal_bottoms = [i for i in items if i.style == "正式" and "裤" in i.category]

    if formal_tops and formal_bottoms:
        outfit_items = formal_tops[:2] + formal_bottoms[:1]
        score = 0.92
        weather_ok = weather.temperature_max >= 10  # 西雅图春季
        outfit_candidates.append(OutfitRecommendation(
            items=outfit_items,
            reason=f"深蓝色西装外套 + 白色衬衫 + 深灰色西裤，专业正式，适合 {intent.occasion}。"
                   f"西雅图当周气温 {weather.temperature_min}~{weather.temperature_max}°C，"
                   f"建议内搭保暖。",
            match_score=score,
            weather_suitable=weather_ok,
            occasion_suitability="非常适合",
        ))

    return outfit_candidates
