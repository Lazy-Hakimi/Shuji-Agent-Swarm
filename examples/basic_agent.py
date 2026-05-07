"""
基本智能体示例
演示如何创建和使用单个智能体
"""
import asyncio
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from shuji.config import get_default_config, PRESET_AGENTS
from shuji.core import ShujiAgent


async def main():
    """主函数"""
    print("=" * 60)
    print("枢机 (Shuji) - 基本智能体示例")
    print("=" * 60)
    
    # 获取配置
    config = get_default_config()
    
    # 使用预设的研究员智能体
    agent_config = PRESET_AGENTS["researcher"]
    
    # 创建智能体
    agent = ShujiAgent(
        agent_id="researcher-001",
        config=agent_config,
    )
    
    print(f"\n智能体已创建: {agent.config.identity.display_name}")
    print(f"角色: {agent.config.role.value}")
    print(f"SOUL配置:\n{agent.config.soul.to_markdown()[:500]}...")
    
    # 模拟对话
    print("\n" + "-" * 60)
    print("开始对话:")
    print("-" * 60)
    
    messages = [
        "你好，请介绍一下你自己",
        "你能帮我做什么？",
        "如何学习人工智能？",
    ]
    
    for message in messages:
        print(f"\n用户: {message}")
        response = await agent.chat(message)
        print(f"智能体: {response}")
    
    # 显示状态
    print("\n" + "-" * 60)
    print("智能体状态:")
    print("-" * 60)
    status = agent.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())