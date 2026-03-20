"""
数据结构定义 — 衣橱管家 Agent
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OccasionType(Enum):
    BUSINESS_FORMAL  = "business_formal"
    TECH_CONFERENCE  = "tech_conference"
    CASUAL           = "casual"
    OUTDOOR          = "outdoor"
    UNKNOWN          = "unknown"


class AgentStep(Enum):
    INIT             = "init"
    INTENT_PARSED    = "intent_parsed"
    CLOTHING_DETECTED = "clothing_detected"
    OUTFIT_RECOMMENDED = "outfit_recommended"
    SHOPPING_DONE    = "shopping_done"
    DONE             = "done"


@dataclass
class UserIntent:
    occasion: str                        # "AI科技峰会"
    location: str                        # "西雅图"
    time_frame: str                      # "下周"
    budget: float                        # 200.0
    currency: str                        # "USD"
    fallback_action: str                 # "网购外套"
    occasion_type: OccasionType = OccasionType.TECH_CONFERENCE


@dataclass
class Detection:
    """传统 CV 检测结果（DINO/YOLO 输出）"""
    det_id: str
    bbox: tuple[float, float, float, float]   # (x, y, w, h) 归一化
    detection_confidence: float               # 检测置信度


@dataclass
class ClothingItem:
    item_id: str
    category: str           # "西装外套" / "衬衫" / "裤子" / ...
    color: str              # "深蓝色"
    style: str              # "正式" / "商务休闲" / "休闲" / "运动"
    material: str           # "羊毛" / "棉" / "uncertain"
    condition: str          # "good" / "wrinkled" / "folded"
    bbox: tuple[float, float, float, float]
    detection_confidence: float
    vlm_agreement_score: float = 1.0  # 多 VLM 投票一致性
    cv_alignment_score: float = 1.0   # CV 分类器与 VLM 对齐度
    final_confidence: float = 0.0     # Layer 3 综合置信度
    conflict_flag: bool = False       # VLM 与 CV 有分歧
    top3_predictions: list[str] = field(default_factory=list)


@dataclass
class WeatherInfo:
    location: str
    date_range: str
    temperature_min: float   # °C
    temperature_max: float
    condition: str           # "Rainy" / "Cloudy" / "Sunny"
    precipitation_prob: float  # 0-1
    wind_speed: float          # km/h


@dataclass
class OutfitRecommendation:
    items: list[ClothingItem]
    reason: str
    match_score: float        # 0-1
    weather_suitable: bool
    occasion_suitability: str


@dataclass
class ShoppingResult:
    product_name: str
    price: float
    currency: str
    url: str
    image_url: str
    match_reason: str
    rating: float


@dataclass
class ConfirmationCard:
    """Layer 4 — 低置信度衣物的交互确认卡片"""
    item_id: str
    ai_guess: str                         # "深蓝色 西装外套"
    alternatives: list[str]               # Top3 备选
    action: str = "tap_to_confirm_or_correct"


@dataclass
class PerceptionResult:
    """AntiHallucinationPipeline 最终输出"""
    confident_items: list[ClothingItem]
    confirmation_cards: list[ConfirmationCard]
    status: str                           # "all_confident" | "needs_confirmation"
    message: str = ""


@dataclass
class AgentState:
    """Orchestrator 贯穿整个 ReAct 循环的状态"""
    step: AgentStep = AgentStep.INIT
    intent: Optional[UserIntent] = None
    weather: Optional[WeatherInfo] = None
    perception: Optional[PerceptionResult] = None
    recommendations: list[OutfitRecommendation] = field(default_factory=list)
    shopping_results: list[ShoppingResult] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # ReAct 轨迹日志
