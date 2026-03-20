"""
AntiHallucinationPipeline — 四层防幻觉体系

Layer 1: Detect-then-Describe  (降低幻觉产生)
Layer 2: Cross-Validation      (交叉校验)
Layer 3: Confidence Calibration (量化不确定性)
Layer 4: Interactive Fallback  (人机协同兜底)
"""
from __future__ import annotations
import asyncio
import uuid
from collections import Counter

from models import (
    ClothingItem, ConfirmationCard, Detection, PerceptionResult
)
import tools


def _iou(b1: tuple, b2: tuple) -> float:
    """计算两个 bbox 的 IoU，格式 (x, y, w, h) 归一化"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _majority_vote(results: list[dict], field: str) -> str:
    """对多个 VLM 输出的某字段做多数投票"""
    votes = [r.get(field, "unknown") for r in results]
    return Counter(votes).most_common(1)[0][0]


def _agreement_score(results: list[dict], field: str) -> float:
    """计算多 VLM 在某字段上的一致度 (0~1)"""
    votes = [r.get(field, "unknown") for r in results]
    most_common_count = Counter(votes).most_common(1)[0][1]
    return most_common_count / len(votes)


CV_CATEGORY_MAP = {
    # cv_category → 中文通用分类（用于对齐检查）
    "blazer": "外套",
    "jacket": "外套",
    "coat": "外套",
    "shirt": "衬衫",
    "t-shirt": "上衣",
    "trousers": "裤子",
    "jeans": "裤子",
    "shorts": "裤子",
    "dress": "连衣裙",
    "sweater": "上衣",
    "hoodie": "上衣",
    "skirt": "裙子",
}


class AntiHallucinationPipeline:
    """
    四层防幻觉 Pipeline。
    输入: 衣橱图片 bytes
    输出: PerceptionResult (confident_items + confirmation_cards)
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    async def run(self, image: bytes) -> PerceptionResult:
        print("\n  [Pipeline] ▶ 开始防幻觉感知流程")

        # Layer 1 → 2 → 3 → 4 串行（每层依赖上层输出）
        items = await self.layer1_perceive(image)
        items = await self.layer2_cross_validate(image, items)
        confident, uncertain = self.layer3_confidence_calibration(items)
        result = self.layer4_interactive_fallback(confident, uncertain)

        print(f"  [Pipeline] ✓ 感知完成: {len(confident)} 件高置信, {len(uncertain)} 件待确认")
        return result

    # ------------------------------------------------------------------
    # Layer 1: Detect-then-Describe
    # ------------------------------------------------------------------
    async def layer1_perceive(self, image: bytes) -> list[ClothingItem]:
        print("\n  [Layer 1] 传统 CV 检测 + 逐件 VLM 识别")

        # Step 1: Grounding-DINO 检测 (确定性，不幻觉)
        detections: list[Detection] = await tools.detect_clothing_regions(image)
        print(f"    DINO 检测到 {len(detections)} 件衣物")

        # Step 2: SAM2 实例分割 → 得到干净的单件裁剪图
        crops: list[bytes] = await tools.segment_clothing(image, detections)

        # Step 3: VLM 对每件单品识别 (输入干净 → 幻觉率低)
        items = []
        for det, crop in zip(detections, crops):
            # 只调用主 VLM (其余在 Layer 2 做多路校验)
            attr = await tools.classify_clothing_vlm(crop, model_id="gpt4o")
            item = ClothingItem(
                item_id=str(uuid.uuid4())[:8],
                category=attr["category"],
                color=attr["color"],
                style=attr["style"],
                material=attr["material"],
                condition="folded" if attr["color"] == "uncertain" else "good",
                bbox=det.bbox,
                detection_confidence=det.detection_confidence,
            )
            items.append(item)
            print(f"    检测: {item.color} {item.category}  (det_conf={det.detection_confidence:.2f})")

        return items

    # ------------------------------------------------------------------
    # Layer 2: Cross-Validation
    # ------------------------------------------------------------------
    async def layer2_cross_validate(
        self, image: bytes, items: list[ClothingItem]
    ) -> list[ClothingItem]:
        print("\n  [Layer 2] 多路交叉校验")

        # 重新获取裁剪图（实际系统会缓存）
        detections = await tools.detect_clothing_regions(image)
        crops = await tools.segment_clothing(image, detections)

        validated = []
        for item, crop in zip(items, crops):
            # 策略1: 3 路 VLM 投票
            vlm_results = await asyncio.gather(
                tools.classify_clothing_vlm(crop, "gpt4o"),
                tools.classify_clothing_vlm(crop, "claude"),
                tools.classify_clothing_vlm(crop, "gemini"),
            )
            category_consensus = _majority_vote(list(vlm_results), "category")
            color_consensus    = _majority_vote(list(vlm_results), "color")
            vlm_agreement      = _agreement_score(list(vlm_results), "category")
            top3 = list({r.get("raw_description", "") for r in vlm_results})

            # 策略2: CV 分类器第二意见
            cv_result = await tools.classify_clothing_cv(crop)
            cv_broad = CV_CATEGORY_MAP.get(cv_result["category"], cv_result["category"])

            # 判断 VLM 分类与 CV 分类是否冲突
            item_broad = "外套" if "外套" in category_consensus or "夹克" in category_consensus else \
                         "裤子" if "裤" in category_consensus else \
                         "衬衫" if "衬衫" in category_consensus else \
                         "上衣"
            conflict = (cv_broad != item_broad) and (vlm_agreement < 0.8)
            cv_align = 1.0 if not conflict else 0.5

            # 策略3: 全局一致性 (简化版: 相似 bbox 去重)
            item.category = category_consensus
            item.color    = color_consensus
            item.vlm_agreement_score = vlm_agreement
            item.cv_alignment_score  = cv_align
            item.conflict_flag       = conflict
            item.top3_predictions    = top3

            if conflict:
                print(f"    ⚠ 冲突: '{item.category}' (VLM) vs '{cv_broad}' (CV)  → 置信度降权")
            else:
                print(f"    ✓ 一致: {item.color} {item.category}  (VLM一致性={vlm_agreement:.2f})")

            validated.append(item)

        # IoU 去重 — 移除重叠度 > 0.5 的重复检测
        deduped = self._deduplicate(validated)
        removed = len(validated) - len(deduped)
        if removed:
            print(f"    去重: 移除 {removed} 个重复检测")

        return deduped

    def _deduplicate(self, items: list[ClothingItem]) -> list[ClothingItem]:
        """基于 bbox IoU 去重，保留置信度更高的"""
        keep = []
        for item in sorted(items, key=lambda x: -x.detection_confidence):
            overlap = any(_iou(item.bbox, k.bbox) > 0.5 for k in keep)
            if not overlap:
                keep.append(item)
        return keep

    # ------------------------------------------------------------------
    # Layer 3: Confidence Calibration
    # ------------------------------------------------------------------
    def layer3_confidence_calibration(
        self, items: list[ClothingItem]
    ) -> tuple[list[ClothingItem], list[ClothingItem]]:
        print("\n  [Layer 3] 置信度校准")

        confident, uncertain = [], []
        for item in items:
            # 综合置信度 = 检测置信度 × VLM一致性 × CV对齐度
            final_conf = (
                item.detection_confidence *
                item.vlm_agreement_score *
                item.cv_alignment_score
            )
            item.final_confidence = round(final_conf, 3)

            bucket = "高置信" if final_conf > self.confidence_threshold else "待确认"
            print(f"    {item.color} {item.category}: "
                  f"det={item.detection_confidence:.2f} × "
                  f"vlm={item.vlm_agreement_score:.2f} × "
                  f"cv={item.cv_alignment_score:.2f} = "
                  f"{final_conf:.3f}  [{bucket}]")

            if final_conf > self.confidence_threshold:
                confident.append(item)
            else:
                uncertain.append(item)

        return confident, uncertain

    # ------------------------------------------------------------------
    # Layer 4: Interactive Fallback
    # ------------------------------------------------------------------
    def layer4_interactive_fallback(
        self,
        confident: list[ClothingItem],
        uncertain: list[ClothingItem],
    ) -> PerceptionResult:
        print("\n  [Layer 4] 交互兜底")

        if not uncertain:
            print("    所有衣物识别置信度达标，无需用户确认")
            return PerceptionResult(
                confident_items=confident,
                confirmation_cards=[],
                status="all_confident",
            )

        cards = []
        for item in uncertain:
            card = ConfirmationCard(
                item_id=item.item_id,
                ai_guess=f"{item.color} {item.category}",
                alternatives=item.top3_predictions[:3],
            )
            cards.append(card)
            print(f"    生成确认卡片: '{card.ai_guess}'  备选: {card.alternatives}")

        # 在真实产品中：推送卡片给前端，等待用户 tap 选择
        # Mock 中：自动接受 AI 猜测，继续流程
        print("    [Mock] 自动接受 AI 猜测，继续处理...")

        # 将 uncertain 也加入最终列表（已标记 conflict_flag=True）
        all_items = confident + uncertain
        return PerceptionResult(
            confident_items=all_items,
            confirmation_cards=cards,
            status="needs_confirmation",
            message=f"我对 {len(uncertain)} 件衣物不太确定，已生成确认卡片（Mock 中自动接受）",
        )
