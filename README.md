# AI 私人衣橱管家

基于多模态 Agent 架构的智能穿搭推荐系统。输入衣橱照片 + 语音指令，自动识别衣物、匹配场合、推荐穿搭，衣橱无合适选项时自动转入网购搜索。

## 快速开始

```bash
python main.py
```

无需安装任何依赖，默认使用 Mock 模式运行完整流程。

---

## 系统架构

```
语音指令 ──→ ASR ──→ IntentParserAgent ──┐
                                          ▼
衣橱照片 ──→ 图像增强 ──→ WardrobeAnalyzerAgent  ──→ OrchestratorAgent (ReAct)
                           AntiHallucinationPipeline        │
                                                     ├──→ StylistAgent
                                                     └──→ ShoppingAgent (fallback)
```

**OrchestratorAgent** 使用 ReAct（Reason + Act）循环统一调度所有子 Agent，每轮输出 Thought / Action / Observation，最终返回结构化推荐结果。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 入口，运行 Demo |
| `models.py` | 全部数据结构定义 |
| `tools.py` | 外部工具接口 + Mock 实现 |
| `pipeline.py` | 四层防幻觉 Pipeline |
| `agents/orchestrator.py` | ReAct 主控循环 |
| `agents/intent_parser.py` | 意图解析 Agent |
| `agents/wardrobe_analyzer.py` | 衣物识别 Agent |
| `agents/stylist.py` | 穿搭推荐 Agent |
| `agents/shopping.py` | 网购搜索 Agent |

---

## 核心模块

### OrchestratorAgent

ReAct 主控循环，最多执行 10 轮，按状态机驱动各子 Agent。

```
INIT → parse_intent → analyze_wardrobe_and_weather → recommend_outfit → finish
                                                    └→ search_shopping → finish
```

**主要方法：**
- `run(image, audio)` — 主入口，返回最终推荐结果 dict
- `_execute(action, text, image)` — 执行单步动作，更新 AgentState
- `_format_result()` — 格式化输出（穿搭方案 or 购物推荐）

---

### AntiHallucinationPipeline

四层防幻觉体系，应对视觉大模型在杂乱场景下的识别误差。

**Layer 1 — Detect-then-Describe**
`layer1_perceive(image)` 先用 Grounding-DINO 做目标检测、SAM2 做实例分割，将衣橱全图切分为单件衣物裁剪图，再用 VLM 对每张干净的单品图做属性识别。避免 VLM 直接分析杂乱全图引发幻觉。

**Layer 2 — Cross-Validation**
`layer2_cross_validate(image, items)` 对每件衣物并行调用 GPT-4o / Claude / Gemini 三路 VLM 投票，同时引入轻量 CV 分类器（EfficientNet）做第二意见。若 VLM 与 CV 结果冲突，则将置信度降权并标记 `conflict_flag`。最后基于 bbox IoU 去除重复检测。

**Layer 3 — Confidence Calibration**
`layer3_confidence_calibration(items)` 计算综合置信度：
```
final_confidence = detection_confidence × vlm_agreement_score × cv_alignment_score
```
低于阈值（默认 0.75）的衣物归入"待确认"组。

**Layer 4 — Interactive Fallback**
`layer4_interactive_fallback(confident, uncertain)` 对不确定衣物生成 `ConfirmationCard`（含裁剪图、AI 猜测、Top3 备选），推送给前端让用户 tap 确认，而非重新描述。Mock 模式下自动接受 AI 猜测。

---

### 外部工具（tools.py）

所有工具均提供真实接口签名和 Mock 实现，通过环境变量切换：

```bash
USE_MOCK=false python main.py  # 接入真实 API
```

| 函数 | 真实依赖 | 说明 |
|------|---------|------|
| `speech_to_text(audio)` | OpenAI Whisper | 语音转文字 |
| `enhance_image(image)` | OpenCV + Real-ESRGAN | CLAHE 增强 + 超分辨率 |
| `detect_clothing_regions(image)` | Grounding-DINO | 衣物目标检测 |
| `segment_clothing(image, detections)` | SAM2 | 实例分割 |
| `classify_clothing_vlm(crop, model_id)` | GPT-4o / Claude / Gemini | 单品属性识别 |
| `classify_clothing_cv(crop)` | EfficientNet | CV 分类器第二意见 |
| `get_weather_forecast(location, date_range)` | OpenWeatherMap API | 天气预报 |
| `search_products(query, max_price)` | Google Shopping / SerpAPI | 电商搜索 |

---

## 数据结构（models.py）

| 类 | 说明 |
|----|------|
| `UserIntent` | 结构化用户意图（场合、地点、预算、兜底策略） |
| `ClothingItem` | 单件衣物（类别、颜色、风格、置信度等） |
| `WeatherInfo` | 天气信息（温度、降水概率、风速） |
| `OutfitRecommendation` | 穿搭方案（衣物列表、匹配度、天气适配性） |
| `ShoppingResult` | 商品推荐（名称、价格、链接、推荐理由） |
| `ConfirmationCard` | 低置信衣物的交互确认卡片 |
| `PerceptionResult` | Pipeline 最终输出（高置信列表 + 确认卡片列表） |
| `AgentState` | Orchestrator 贯穿 ReAct 循环的完整状态 |
