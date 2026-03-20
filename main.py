"""
main.py — AI 衣橱管家 Demo 入口

使用方式:
    python main.py           # Mock 模式（默认）
    USE_MOCK=false python main.py  # 真实 API（需配置各 API Key）
"""
import asyncio
import sys
import os

# 将当前目录加入 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from agents.orchestrator import OrchestratorAgent


async def demo():
    # --- Mock 输入 ---
    # 真实场景: 读取用户上传的图片文件和录音文件
    mock_image = b"MOCK_WARDROBE_IMAGE_BYTES_1024x768"
    mock_audio = b"MOCK_AUDIO_BYTES_WAV_FORMAT"

    agent = OrchestratorAgent()
    result = await agent.run(image=mock_image, audio=mock_audio)

    print("\n" + "=" * 60)
    print("   结构化返回结果 (JSON)")
    print("=" * 60)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(demo())
