"""
多智能体团队示例
演示如何创建和协调多个智能体
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shuji.config import PRESET_AGENTS
from shuji.core import ShujiAgent, AgentTeam
from shuji.skills import SkillManager


async def main():
    """主函数"""
    print("=" * 60)
    print("枢机 (Shuji) - 多智能体团队示例")
    print("=" * 60)
    
    # 创建技能管理器
    skill_manager = SkillManager()
    skill_manager.register_builtin_skills()
    
    print(f"\n可用技能: {[s['name'] for s in skill_manager.list_skills()]}")
    
    # 创建智能体
    researcher = ShujiAgent(
        agent_id="researcher-001",
        config=PRESET_AGENTS["researcher"],
    )
    
    writer = ShujiAgent(
        agent_id="writer-001",
        config=PRESET_AGENTS["writer"],
    )
    
    coder = ShujiAgent(
        agent_id="coder-001",
        config=PRESET_AGENTS["coder"],
    )
    
    # 注册技能
    for agent in [researcher, writer, coder]:
        agent.register_tool("file", skill_manager.execute)
        agent.register_tool("search", skill_manager.execute)
    
    print("\n智能体团队:")
    for agent in [researcher, writer, coder]:
        print(f"  - {agent.config.identity.display_name} ({agent.config.role.value})")
    
    # 创建顺序执行团队
    print("\n" + "-" * 60)
    print("顺序执行模式:")
    print("-" * 60)
    
    sequential_team = AgentTeam(
        agents=[researcher, writer, coder],
        coordination_mode="sequential",
        shared_memory=True,
    )
    
    print(f"\n团队状态: {sequential_team.get_status()}")
    
    # 模拟任务执行
    task = "研究Python异步编程并撰写一篇教程"
    print(f"\n任务: {task}")
    print("执行中...")
    
    # 注意：实际执行需要模型支持，这里仅演示API
    # result = await sequential_team.execute(task)
    # print(f"结果:\n{result}")
    
    # 创建并行执行团队
    print("\n" + "-" * 60)
    print("并行执行模式:")
    print("-" * 60)
    
    parallel_team = AgentTeam(
        agents=[researcher, writer],
        coordination_mode="parallel",
        shared_memory=True,
    )
    
    print(f"\n团队状态: {parallel_team.get_status()}")
    
    # 层级执行团队 (需要协调者)
    print("\n" + "-" * 60)
    print("层级执行模式 (Orchestrator):")
    print("-" * 60)
    
    orchestrator = ShujiAgent(
        agent_id="orchestrator-001",
        config=PRESET_AGENTS.get("orchestrator", PRESET_AGENTS["researcher"]),
    )
    
    hierarchical_team = AgentTeam(
        agents=[orchestrator, researcher, writer, coder],
        coordination_mode="hierarchical",
        shared_memory=True,
    )
    
    print(f"\n团队状态: {hierarchical_team.get_status()}")


if __name__ == "__main__":
    asyncio.run(main())