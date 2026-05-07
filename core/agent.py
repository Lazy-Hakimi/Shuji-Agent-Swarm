"""
枢机 (Shuji) - Agent智能体系统
实现基于DeepSeekV3Mini的智能体
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re

import torch

from ..config import AgentConfig, AgentIdentity, SoulConfig, AgentRole, AgentStatus
from ..models import ShujiForCausalLM
from ..memory import MemorySystem
from ..protocols.mcp import MCPClient
from ..protocols.acp import ACPClient


class AgentState(Enum):
    """Agent状态机状态"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class AgentMessage:
    """Agent消息"""
    role: str  # system, user, assistant, tool
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AgentTask:
    """Agent任务"""
    task_id: str
    description: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    plan: Optional[List[Dict]] = None  # 任务计划


class ShujiAgent:
    """
    枢机智能体
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        model: Optional[ShujiForCausalLM] = None,
        tokenizer=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        memory_system: Optional[MemorySystem] = None,
    ):
        self.agent_id = agent_id
        self.config = config
        self.device = device
        self.logger = logging.getLogger(f"ShujiAgent.{agent_id}")
        
        # 模型
        self.model = model
        self.tokenizer = tokenizer
        self._load_model_if_needed()
        
        # 状态
        self.state = AgentState.IDLE
        self.status = AgentStatus.IDLE
        
        # 消息历史
        self.messages: List[AgentMessage] = []
        self.max_messages = 100
        
        # 任务队列
        self.tasks: Dict[str, AgentTask] = {}
        self.current_task: Optional[AgentTask] = None
        
        # 工具注册
        self.tools: Dict[str, Callable] = {}
        
        # 记忆系统
        self.memory = memory_system
        
        # 协议客户端（可选）
        self.mcp_client: Optional[MCPClient] = None
        self.acp_client: Optional[ACPClient] = None
        
        # 统计
        self.stats = {
            "total_messages": 0,
            "total_tasks": 0,
            "created_at": time.time(),
            "last_activity": time.time(),
        }
        
        # 初始化系统消息
        self._init_system_message()
    
    def _load_model_if_needed(self):
        """如果需要，加载模型和分词器"""
        if self.model is None:
            # 实际项目中应从配置加载预训练模型
            # 此处留空，由外部注入
            self.logger.warning("No model provided, agent will use mock responses")
    
    def _init_system_message(self):
        """初始化系统消息"""
        soul = self.config.soul
        system_prompt = f"""You are {soul.name}, {soul.role}.

Personality:
{chr(10).join(f"- {trait}" for trait in soul.traits)}

Communication Style: {soul.communication_style}
Tone: {soul.tone}

Rules:
{chr(10).join(f"- {rule}" for rule in soul.rules)}

Prohibitions:
{chr(10).join(f"- {p}" for p in soul.prohibitions)}

You have access to the following tools: {', '.join(self.config.enabled_tools)}

Respond in {soul.preferred_language} using {soul.output_format} format.
"""
        
        self.messages.append(AgentMessage(
            role="system",
            content=system_prompt
        ))
    
    def register_tool(self, name: str, handler: Callable):
        """注册工具"""
        self.tools[name] = handler
        self.logger.debug(f"Registered tool: {name}")
    
    def unregister_tool(self, name: str):
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
    
    async def chat(self, message: str, context: Optional[Dict] = None) -> str:
        """
        与Agent对话
        """
        self.state = AgentState.THINKING
        self.status = AgentStatus.BUSY
        
        try:
            # 添加用户消息
            self.add_message("user", message, context)
            
            # 检索相关记忆
            memories = []
            if self.memory:
                memories = await self.memory.search(message, k=3)
            
            # 构建提示
            prompt = self._build_prompt(message, memories)
            
            # 生成回复
            response = await self._generate_response(prompt)
            
            # 添加助手消息
            self.add_message("assistant", response)
            
            # 保存到记忆
            if self.memory:
                await self.memory.add(f"User: {message}\nAssistant: {response}")
            
            self.state = AgentState.IDLE
            self.status = AgentStatus.IDLE
            self.stats["last_activity"] = time.time()
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            self.state = AgentState.ERROR
            self.status = AgentStatus.ERROR
            raise
    
    async def run_task(self, task_description: str, tools: Optional[List[str]] = None) -> str:
        """
        运行任务
        """
        task_id = str(uuid.uuid4())
        task = AgentTask(
            task_id=task_id,
            description=task_description,
        )
        self.tasks[task_id] = task
        self.current_task = task
        self.stats["total_tasks"] += 1
        
        task.status = "running"
        task.started_at = time.time()
        self.state = AgentState.EXECUTING
        
        try:
            # 任务规划
            plan = await self._plan_task(task_description)
            task.plan = plan
            
            # 执行任务步骤
            result = await self._execute_plan(plan, tools)
            
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            
            self.state = AgentState.IDLE
            self.status = AgentStatus.IDLE
            
            return result
        
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.state = AgentState.ERROR
            self.logger.error(f"Task failed: {e}")
            raise
        finally:
            self.current_task = None
    
    async def _plan_task(self, task_description: str) -> List[Dict]:
        """规划任务 - 使用模型生成计划"""
        prompt = f"""Task: {task_description}

You are an AI assistant that breaks down tasks into a sequence of steps. 
For each step, output a JSON object with:
- "action": description of the step
- "tools": list of tool names needed (from: {', '.join(self.config.enabled_tools)})
- "expected_output": what this step should produce

Return a JSON array of steps. Example:
[
  {{"action": "Search for recent AI papers", "tools": ["search"], "expected_output": "list of papers"}},
  {{"action": "Summarize each paper", "tools": [], "expected_output": "summaries"}}
]

Now output the JSON array for the given task:"""
        
        response = await self._generate_response(prompt)
        
        # 尝试从响应中提取JSON数组
        plan = self._extract_json_array(response)
        if plan is not None:
            return plan
        else:
            # 降级：返回简单计划
            self.logger.warning("Failed to parse plan JSON, using fallback plan")
            return [{"action": task_description, "tools": [], "expected_output": "task result"}]
    
    def _extract_json_array(self, text: str) -> Optional[List[Dict]]:
        """从文本中提取JSON数组"""
        # 尝试匹配 ```json ... ``` 块
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text)
        if match:
            json_str = match.group(1)
        else:
            # 尝试匹配 [ ... ] 块
            array_pattern = r'(\[[\s\S]*\])'
            match = re.search(array_pattern, text)
            if match:
                json_str = match.group(1)
            else:
                json_str = text.strip()
        
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return None
    
    async def _execute_plan(self, plan: List[Dict], allowed_tools: Optional[List[str]]) -> str:
        """执行计划"""
        results = []
        
        for step in plan:
            action = step.get("action", "")
            tools_needed = step.get("tools", [])
            
            # 执行工具调用
            step_tool_results = []
            for tool_name in tools_needed:
                if allowed_tools and tool_name not in allowed_tools:
                    continue
                if tool_name in self.tools:
                    try:
                        # 调用工具，传递action作为参数（可根据需要调整）
                        tool_result = await self.tools[tool_name](action=action, **step.get("params", {}))
                        step_tool_results.append(f"Tool '{tool_name}' result: {tool_result}")
                    except Exception as e:
                        step_tool_results.append(f"Tool '{tool_name}' error: {e}")
                else:
                    step_tool_results.append(f"Tool '{tool_name}' not available")
            
            # 生成步骤总结
            step_prompt = f"""Action: {action}
Tool results: {chr(10).join(step_tool_results) if step_tool_results else 'No tools used'}

Provide a concise summary of this step."""
            
            step_result = await self._generate_response(step_prompt)
            results.append(f"Step: {action}\nResult: {step_result}")
        
        # 最终总结
        final_prompt = f"""Task: {plan[0].get('action', 'Task') if plan else 'Task'}

Step results:
{chr(10).join(results)}

Provide a final overall result of the task."""
        
        final_result = await self._generate_response(final_prompt)
        return final_result
    
    async def _generate_response(self, prompt: str) -> str:
        """生成回复"""
        if self.model is None or self.tokenizer is None:
            # 模拟回复（开发测试用）
            self.logger.debug(f"Using mock response for prompt: {prompt[:100]}...")
            return f"[Mock response for: {prompt[:50]}...]"
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
            )
        
        # Decode
        response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return response
    
    def _build_prompt(self, message: str, memories: List[str]) -> str:
        """构建提示"""
        # 添加记忆上下文
        memory_context = ""
        if memories:
            memory_context = "Relevant memories:\n" + "\n".join(memories) + "\n\n"
        
        # 添加对话历史
        history = ""
        for msg in self.messages[-self.config.memory_max_context:]:
            if msg.role != "system":
                history += f"{msg.role.capitalize()}: {msg.content}\n"
        
        prompt = f"""{memory_context}{history}
User: {message}
Assistant:"""
        
        return prompt
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息"""
        self.messages.append(AgentMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        
        # 限制消息数量
        if len(self.messages) > self.max_messages:
            # 保留系统消息和最近的消息
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"][-(self.max_messages - len(system_msgs)):]
            self.messages = system_msgs + other_msgs
        
        self.stats["total_messages"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "agent_id": self.agent_id,
            "name": self.config.identity.name,
            "state": self.state.value,
            "status": self.status.value,
            "role": self.config.role.value,
            "current_task": self.current_task.task_id if self.current_task else None,
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "total_messages": self.stats["total_messages"],
            "total_tasks": self.stats["total_tasks"],
            "uptime": time.time() - self.stats["created_at"],
            "last_activity": self.stats["last_activity"],
        }
    
    def pause(self):
        """暂停Agent"""
        self.state = AgentState.PAUSED
        self.status = AgentStatus.PAUSED
    
    def resume(self):
        """恢复Agent"""
        self.state = AgentState.IDLE
        self.status = AgentStatus.IDLE
    
    def reset(self):
        """重置Agent"""
        self.messages = []
        self.tasks = {}
        self.current_task = None
        self.state = AgentState.IDLE
        self.status = AgentStatus.IDLE
        self._init_system_message()


class AgentManager:
    """Agent管理器"""
    
    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, ShujiAgent] = {}
        self.logger = logging.getLogger("AgentManager")
        self._load_default_model()
    
    def _load_default_model(self):
        """加载默认模型（可选）"""
        # 实际项目中可加载预训练模型
        self.default_model = None
        self.default_tokenizer = None
    
    def create_agent(
        self,
        agent_id: Optional[str] = None,
        agent_config: Optional[AgentConfig] = None,
        model: Optional[ShujiForCausalLM] = None,
        tokenizer=None,
        memory_system: Optional[MemorySystem] = None,
    ) -> ShujiAgent:
        """创建Agent"""
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        
        if agent_config is None:
            from ..config import PRESET_AGENTS
            agent_config = PRESET_AGENTS.get("generalist", AgentConfig())
        
        agent = ShujiAgent(
            agent_id,
            agent_config,
            model or self.default_model,
            tokenizer or self.default_tokenizer,
            memory_system=memory_system,
        )
        self.agents[agent_id] = agent
        
        self.logger.info(f"Created agent: {agent_id}")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[ShujiAgent]:
        """获取Agent"""
        return self.agents.get(agent_id)
    
    def delete_agent(self, agent_id: str):
        """删除Agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Deleted agent: {agent_id}")
    
    def list_agents(self) -> List[Dict]:
        """列出所有Agent"""
        return [agent.get_status() for agent in self.agents.values()]
    
    # Gateway handlers
    async def handle_chat(self, connection, params: Dict) -> Dict:
        """处理聊天请求"""
        agent_id = params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing agent_id")
        message = params.get("message")
        if not message:
            raise ValueError("Missing message")
        
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        response = await agent.chat(message)
        return {"response": response}
    
    async def handle_run(self, connection, params: Dict) -> Dict:
        """处理运行任务请求"""
        agent_id = params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing agent_id")
        task = params.get("task")
        if not task:
            raise ValueError("Missing task")
        
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        result = await agent.run_task(task)
        return {"result": result}
    
    async def handle_status(self, connection, params: Dict) -> Dict:
        """处理状态请求"""
        agent_id = params.get("agent_id")
        
        if agent_id:
            agent = self.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")
            return agent.get_status()
        else:
            return {"agents": self.list_agents()}
    
    async def handle_list(self, connection, params: Dict) -> Dict:
        """处理列表请求"""
        return {"agents": self.list_agents()}
    
    async def handle_create(self, connection, params: Dict) -> Dict:
        """处理创建请求"""
        agent_type = params.get("type", "generalist")
        
        from ..config import PRESET_AGENTS
        if agent_type in PRESET_AGENTS:
            config = PRESET_AGENTS[agent_type]
        else:
            config = AgentConfig()
        
        agent = self.create_agent(agent_config=config)
        return {"agent_id": agent.agent_id, "status": "created"}
    
    async def handle_delete(self, connection, params: Dict) -> Dict:
        """处理删除请求"""
        agent_id = params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing agent_id")
        
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        self.delete_agent(agent_id)
        return {"agent_id": agent_id, "status": "deleted"}