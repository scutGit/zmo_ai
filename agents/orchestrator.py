"""
OrchestratorAgent — 主控 Agent，ReAct (Reason + Act) 循环

ReAct 轨迹格式:
  Thought: 分析当前状态，决定下一步
  Action:  调用某个子 Agent 或工具
  Observation: 获得结果
  ... 重复 ...
  Final Answer: 汇总输出
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional

from models import AgentState, AgentStep, UserIntent
import tools


# ---------------------------------------------------------------------------
# ReAct 轨迹节点
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    thought: str
    action: str
    observation: str


# ---------------------------------------------------------------------------
# Mock LLM — 按 state.step 返回确定性的下一步动作
# ---------------------------------------------------------------------------

def _mock_llm_decide(state: AgentState) -> tuple[str, str]:
    """
    返回 (thought, action_name)
    真实版本: 调用 LLM chat completions，解析 tool_use / end_turn
    """
    step = state.step

    if step == AgentStep.INIT:
        return (
            "用户发来了语音指令和衣橱照片。首先应并行预处理：ASR 转文字 + 图像增强。"
            "然后解析用户意图，了解场合、地点、预算。",
            "parse_intent"
        )
    elif step == AgentStep.INTENT_PARSED:
        intent = state.intent
        return (
            f"已解析意图：场合={intent.occasion}，地点={intent.location}，"
            f"预算={intent.budget}{intent.currency}。"
            "下一步：用防幻觉 Pipeline 识别衣橱中的所有衣物，同时查询天气预报。",
            "analyze_wardrobe_and_weather"
        )
    elif step == AgentStep.CLOTHING_DETECTED:
        items = state.perception.confident_items
        formal = [i for i in items if i.style == "正式"]
        if len(formal) >= 2:
            return (
                f"识别到 {len(items)} 件衣物，其中 {len(formal)} 件正式风格。"
                "天气已获取。尝试为用户推荐合适的穿搭方案。",
                "recommend_outfit"
            )
        else:
            return (
                f"识别到 {len(items)} 件衣物，正式风格衣物不足，无法组成完整峰会穿搭。"
                "按 fallback 策略转入网络购物搜索。",
                "search_shopping"
            )
    elif step == AgentStep.OUTFIT_RECOMMENDED:
        recs = state.recommendations
        if recs and recs[0].match_score >= 0.8:
            return (
                f"穿搭推荐成功，最高匹配度 {recs[0].match_score:.0%}，无需购物。"
                "任务完成，准备输出结果。",
                "finish"
            )
        else:
            return (
                "衣橱穿搭匹配度不够理想，启动网络购物补充。",
                "search_shopping"
            )
    elif step == AgentStep.SHOPPING_DONE:
        return (
            "购物搜索完成。任务完成，准备输出最终结果。",
            "finish"
        )
    else:
        return ("任务已完成。", "finish")


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class OrchestratorAgent:

    MAX_ITERATIONS = 10

    def __init__(self):
        from agents.intent_parser import IntentParserAgent
        from agents.wardrobe_analyzer import WardrobeAnalyzerAgent
        from agents.stylist import StylistAgent
        from agents.shopping import ShoppingAgent

        self.intent_parser    = IntentParserAgent()
        self.wardrobe_analyzer = WardrobeAnalyzerAgent()
        self.stylist          = StylistAgent()
        self.shopping         = ShoppingAgent()

        self.state = AgentState()
        self.trace: list[TraceEntry] = []

    async def run(self, image: bytes, audio: bytes) -> dict:
        print("=" * 60)
        print("   AI 衣橱管家 — OrchestratorAgent 启动")
        print("=" * 60)

        # Step 0: 并行预处理 — ASR + 图像增强
        print("\n[预处理] 并行执行: 语音转文字 + 图像增强...")
        text, enhanced_image = await asyncio.gather(
            tools.speech_to_text(audio),
            tools.enhance_image(image),
        )
        print(f"[预处理] ✓ 转录文本: \"{text[:60]}...\"")
        print(f"[预处理] ✓ 图像增强完成")

        # 将输入存入 state
        self._log("预处理完成", "speech_to_text + enhance_image",
                  f"文本: {text[:50]}..., 图像: {len(enhanced_image)} bytes")

        # 进入 ReAct 循环
        for iteration in range(self.MAX_ITERATIONS):
            print(f"\n{'─'*60}")
            print(f"  ReAct 循环 第 {iteration + 1} 轮  (当前步骤: {self.state.step.value})")
            print(f"{'─'*60}")

            # Reason: Mock LLM 决策
            thought, action = _mock_llm_decide(self.state)
            print(f"  Thought: {thought}")
            print(f"  Action:  {action}")

            if action == "finish":
                self._log(thought, action, "任务完成")
                break

            # Act: 执行动作
            observation = await self._execute(action, text, enhanced_image)
            print(f"  Observation: {observation[:120]}")

            # 记录轨迹
            self._log(thought, action, observation)

        print(f"\n{'='*60}")
        print("   最终结果")
        print(f"{'='*60}")
        return self._format_result()

    async def _execute(self, action: str, text: str, image: bytes) -> str:
        """执行具体动作，更新 state"""

        if action == "parse_intent":
            intent = await self.intent_parser.parse(text)
            self.state.intent = intent
            self.state.step = AgentStep.INTENT_PARSED
            return (f"意图解析成功: 场合={intent.occasion}, 地点={intent.location}, "
                    f"时间={intent.time_frame}, 预算={intent.budget}{intent.currency}, "
                    f"兜底策略={intent.fallback_action}")

        elif action == "analyze_wardrobe_and_weather":
            # 并行: 衣物识别 + 天气查询
            perception, weather = await asyncio.gather(
                self.wardrobe_analyzer.analyze(image),
                tools.get_weather_forecast(
                    self.state.intent.location,
                    self.state.intent.time_frame,
                ),
            )
            self.state.perception = perception
            self.state.weather    = weather
            self.state.step       = AgentStep.CLOTHING_DETECTED
            items_summary = ", ".join(
                f"{i.color}{i.category}" for i in perception.confident_items[:4]
            )
            return (f"衣物识别完成: {len(perception.confident_items)} 件 ({items_summary}...)。"
                    f"天气: {weather.condition}, "
                    f"{weather.temperature_min}~{weather.temperature_max}°C, "
                    f"降水概率 {weather.precipitation_prob:.0%}")

        elif action == "recommend_outfit":
            recs = await self.stylist.recommend(
                self.state.perception.confident_items,
                self.state.intent,
                self.state.weather,
            )
            self.state.recommendations = recs
            self.state.step = AgentStep.OUTFIT_RECOMMENDED
            if recs:
                best = recs[0]
                names = " + ".join(f"{i.color}{i.category}" for i in best.items)
                return f"推荐穿搭: {names} (匹配度={best.match_score:.0%}, 理由: {best.reason[:60]})"
            else:
                return "未找到合适穿搭方案，建议转入购物搜索。"

        elif action == "search_shopping":
            results = await self.shopping.search(self.state.intent)
            self.state.shopping_results = results
            self.state.step = AgentStep.SHOPPING_DONE
            summary = "; ".join(f"{r.product_name}(${r.price})" for r in results[:3])
            return f"购物搜索完成: {summary}"

        else:
            return f"未知动作: {action}"

    def _log(self, thought: str, action: str, observation: str):
        self.trace.append(TraceEntry(thought, action, observation))
        self.state.history.append({"thought": thought, "action": action, "obs": observation})

    def _format_result(self) -> dict:
        result = {
            "status": "success",
            "react_steps": len(self.trace),
            "clothing_detected": len(self.state.perception.confident_items) if self.state.perception else 0,
        }

        if self.state.recommendations:
            best = self.state.recommendations[0]
            outfit_items = [f"{i.color}{i.category}" for i in best.items]
            result["recommendation_type"] = "wardrobe"
            result["outfit"] = outfit_items
            result["match_score"] = f"{best.match_score:.0%}"
            result["weather_suitable"] = best.weather_suitable
            result["reason"] = best.reason

            print(f"\n✅ 推荐类型: 从衣橱中搭配")
            print(f"   穿搭方案: {' + '.join(outfit_items)}")
            print(f"   匹配度: {best.match_score:.0%}")
            print(f"   适应天气: {'是' if best.weather_suitable else '否'}")
            print(f"   理由: {best.reason}")

        elif self.state.shopping_results:
            top = self.state.shopping_results[0]
            result["recommendation_type"] = "shopping"
            result["product"] = top.product_name
            result["price"] = f"{top.price} {top.currency}"
            result["url"] = top.url
            result["match_reason"] = top.match_reason

            print(f"\n🛍 推荐类型: 网购补充")
            for r in self.state.shopping_results:
                print(f"   • {r.product_name}: ${r.price} ★{r.rating}")
                print(f"     理由: {r.match_reason}")
        else:
            result["recommendation_type"] = "none"
            print("\n❌ 未能生成推荐")

        if self.state.perception and self.state.perception.confirmation_cards:
            cards = self.state.perception.confirmation_cards
            result["confirmation_needed"] = len(cards)
            print(f"\n⚠ 有 {len(cards)} 件衣物需要用户确认:")
            for c in cards:
                print(f"   [{c.item_id}] AI猜测: '{c.ai_guess}'  备选: {c.alternatives}")

        return result
