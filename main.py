"""
枢机 (Shuji) - 主入口
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config, get_lightweight_config
from core import GatewayServer, AgentManager
from memory import MemorySystem
from skills import SkillManager


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


async def start_gateway(config):
    """启动网关"""
    # 创建组件
    memory_system = MemorySystem(config)
    await memory_system.initialize()
    
    skill_manager = SkillManager(config.skills_directory)
    skill_manager.register_builtin_skills()
    
    agent_manager = AgentManager(config)
    
    # 创建网关服务器
    server = GatewayServer(config)
    server.register_agent_handler(agent_manager)
    
    # 启动
    await server.run_forever()


async def create_agent(config, agent_type: str = "generalist"):
    """创建智能体"""
    from config import PRESET_AGENTS
    
    agent_manager = AgentManager(config)
    
    if agent_type in PRESET_AGENTS:
        agent_config = PRESET_AGENTS[agent_type]
    else:
        from config import AgentConfig
        agent_config = AgentConfig()
    
    agent = agent_manager.create_agent(agent_config=agent_config)
    
    print(f"Created agent: {agent.agent_id}")
    print(f"Name: {agent.config.identity.name}")
    print(f"Role: {agent.config.role.value}")
    
    return agent


async def interactive_mode(config):
    """交互模式"""
    print("=" * 60)
    print("枢机 (Shuji) - 交互模式")
    print("=" * 60)
    
    # 创建智能体
    agent_manager = AgentManager(config)
    agent = agent_manager.create_agent()
    
    print(f"\n智能体已创建: {agent.config.identity.display_name}")
    print("输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            if not user_input:
                continue
            
            # 获取回复
            response = await agent.chat(user_input)
            print(f"\n{agent.config.identity.display_name}: {response}\n")
        
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='枢机 (Shuji) - 多智能体框架')
    parser.add_argument(
        'command',
        choices=['gateway', 'agent', 'interactive', 'demo'],
        help='要执行的命令'
    )
    parser.add_argument(
        '--config',
        choices=['default', 'lightweight', 'enterprise'],
        default='default',
        help='配置类型'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    parser.add_argument(
        '--agent-type',
        default='generalist',
        help='智能体类型'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 获取配置
    if args.config == 'default':
        config = get_default_config()
    elif args.config == 'lightweight':
        config = get_lightweight_config()
    else:
        from config import get_enterprise_config
        config = get_enterprise_config()
    
    # 执行命令
    if args.command == 'gateway':
        print(f"Starting Shuji Gateway on {config.gateway_host}:{config.gateway_port}")
        asyncio.run(start_gateway(config))
    
    elif args.command == 'agent':
        asyncio.run(create_agent(config, args.agent_type))
    
    elif args.command == 'interactive':
        asyncio.run(interactive_mode(config))
    
    elif args.command == 'demo':
        print("=" * 60)
        print("枢机 (Shuji) - 演示模式")
        print("=" * 60)
        print(f"\n模型配置:")
        print(f"  总参数量: {config.total_params / 1e6:.2f}M")
        print(f"  激活参数量: {config.activated_params / 1e6:.2f}M")
        print(f"  层数: {config.num_layers}")
        print(f"  隐藏维度: {config.hidden_size}")
        print(f"  注意力头数: {config.num_attention_heads}")
        print(f"\n网关配置:")
        print(f"  主机: {config.gateway_host}")
        print(f"  端口: {config.gateway_port}")
        print(f"\n系统已就绪!")


if __name__ == '__main__':
    main()