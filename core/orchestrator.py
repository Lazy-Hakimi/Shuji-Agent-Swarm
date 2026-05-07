"""
枢机 (Shuji) - 多智能体协调器
实现Hub-and-Spoke编排架构
"""
import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from .agent import ShujiAgent, AgentState


class CoordinationMode(Enum):
    """协调模式"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


@dataclass
class Task:
    """任务定义"""
    task_id: str
    description: str
    assigned_agent: Optional[str] = None
    dependencies: List[str] = None
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class Workflow:
    """工作流定义"""
    workflow_id: str
    name: str
    tasks: Dict[str, Task]
    mode: CoordinationMode = CoordinationMode.SEQUENTIAL


class AgentTeam:
    """
    智能体团队
    """
    
    def __init__(
        self,
        agents: List[ShujiAgent],
        coordination_mode: CoordinationMode = CoordinationMode.SEQUENTIAL,
        shared_memory: bool = True,
    ):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.coordination_mode = coordination_mode
        self.shared_memory = shared_memory
        self.logger = logging.getLogger("AgentTeam")
        
        # 任务队列
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, Task] = {}
        
        # 共享上下文
        self.shared_context: Dict[str, Any] = {}
    
    def add_agent(self, agent: ShujiAgent):
        """添加智能体"""
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: str):
        """移除智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    async def execute(self, task_description: str) -> str:
        """
        执行任务
        """
        if self.coordination_mode == CoordinationMode.SEQUENTIAL:
            return await self._execute_sequential(task_description)
        elif self.coordination_mode == CoordinationMode.PARALLEL:
            return await self._execute_parallel(task_description)
        elif self.coordination_mode == CoordinationMode.HIERARCHICAL:
            return await self._execute_hierarchical(task_description)
        else:
            raise ValueError(f"Unknown coordination mode: {self.coordination_mode}")
    
    async def _execute_sequential(self, task_description: str) -> str:
        """顺序执行"""
        results = []
        
        for agent_id, agent in self.agents.items():
            self.logger.info(f"Executing with agent: {agent.config.identity.name}")
            
            # 构建上下文
            context = self._build_context(task_description, results)
            
            # 执行
            result = await agent.run_task(context)
            results.append(f"{agent.config.identity.name}: {result}")
            
            # 更新共享上下文
            if self.shared_memory:
                self.shared_context[agent_id] = result
        
        return "\n\n".join(results)
    
    async def _execute_parallel(self, task_description: str) -> str:
        """并行执行"""
        # 创建任务
        tasks = []
        for agent_id, agent in self.agents.items():
            task = agent.run_task(task_description)
            tasks.append(task)
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        output = []
        for (agent_id, agent), result in zip(self.agents.items(), results):
            if isinstance(result, Exception):
                output.append(f"{agent.config.identity.name}: Error - {result}")
            else:
                output.append(f"{agent.config.identity.name}: {result}")
                if self.shared_memory:
                    self.shared_context[agent_id] = result
        
        return "\n\n".join(output)
    
    async def _execute_hierarchical(self, task_description: str) -> str:
        """层级执行 (Orchestrator模式)"""
        # 找到协调者
        orchestrator = None
        workers = {}
        
        for agent_id, agent in self.agents.items():
            if agent.config.role.value == "orchestrator":
                orchestrator = agent
            else:
                workers[agent_id] = agent
        
        if orchestrator is None:
            # 如果没有协调者，使用第一个智能体
            orchestrator = list(self.agents.values())[0]
            workers = {k: v for k, v in self.agents.items() if k != orchestrator.agent_id}
        
        # 协调者分解任务
        self.logger.info(f"Orchestrator {orchestrator.config.identity.name} planning task")
        
        plan_prompt = f"""Task: {task_description}

Available workers: {', '.join(w.config.identity.name for w in workers.values())}

Break this task into subtasks and assign each to the appropriate worker.
Return a JSON array of subtasks with 'worker', 'description', and 'order' fields.
Example:
[
  {{"worker": "researcher", "description": "Research topic", "order": 1}},
  {{"worker": "writer", "description": "Write based on research", "order": 2}}
]

Now output the JSON array:"""
        
        plan_response = await orchestrator.chat(plan_prompt)
        
        # 解析计划
        subtasks = self._parse_plan(plan_response)
        
        # 按order排序
        subtasks.sort(key=lambda x: x.get("order", 0))
        
        # 执行子任务
        results = []
        for subtask in subtasks:
            worker_name = subtask.get("worker", "")
            description = subtask.get("description", "")
            
            # 找到对应的worker（通过名称匹配）
            worker = None
            for w in workers.values():
                if worker_name.lower() in w.config.identity.name.lower():
                    worker = w
                    break
            
            if worker:
                self.logger.info(f"Executing subtask with {worker.config.identity.name}: {description}")
                result = await worker.run_task(description)
                results.append(f"{worker.config.identity.name}: {result}")
            else:
                self.logger.warning(f"No worker found for: {worker_name}")
                results.append(f"No worker found for: {worker_name}")
        
        # 协调者汇总结果
        summary_prompt = f"""Original task: {task_description}

Subtask results:
{chr(10).join(results)}

Provide a final summary that integrates all results."""
        
        final_result = await orchestrator.chat(summary_prompt)
        
        return f"{final_result}\n\nDetails:\n" + "\n".join(results)
    
    def _build_context(self, task_description: str, previous_results: List[str]) -> str:
        """构建上下文"""
        if not previous_results:
            return task_description
        
        context = f"Task: {task_description}\n\nPrevious results:\n"
        for result in previous_results:
            context += f"- {result}\n"
        context += "\nContinue based on previous results."
        
        return context
    
    def _parse_plan(self, plan_text: str) -> List[Dict]:
        """解析计划文本为JSON数组"""
        # 尝试提取JSON数组
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, plan_text)
        if match:
            json_str = match.group(1)
        else:
            # 尝试直接匹配数组
            array_pattern = r'(\[[\s\S]*\])'
            match = re.search(array_pattern, plan_text)
            if match:
                json_str = match.group(1)
            else:
                json_str = plan_text.strip()
        
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse plan JSON, using fallback")
            # 降级：返回一个简单的子任务
            return [{"worker": "worker", "description": plan_text, "order": 1}]
        
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """获取团队状态"""
        return {
            "agent_count": len(self.agents),
            "coordination_mode": self.coordination_mode.value,
            "agents": {
                agent_id: {
                    "name": agent.config.identity.name,
                    "role": agent.config.role.value,
                    "state": agent.state.value,
                }
                for agent_id, agent in self.agents.items()
            },
        }


class AgentOrchestrator:
    """
    智能体编排器
    """
    
    def __init__(self):
        self.teams: Dict[str, AgentTeam] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.logger = logging.getLogger("AgentOrchestrator")
    
    def create_team(
        self,
        team_id: str,
        agents: List[ShujiAgent],
        mode: CoordinationMode = CoordinationMode.SEQUENTIAL,
    ) -> AgentTeam:
        """创建团队"""
        team = AgentTeam(agents=agents, coordination_mode=mode)
        self.teams[team_id] = team
        return team
    
    def create_workflow(
        self,
        name: str,
        tasks: List[Task],
        mode: CoordinationMode = CoordinationMode.SEQUENTIAL,
    ) -> Workflow:
        """创建工作流"""
        workflow_id = str(uuid.uuid4())
        task_dict = {task.task_id: task for task in tasks}
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            tasks=task_dict,
            mode=mode,
        )
        
        self.workflows[workflow_id] = workflow
        return workflow
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, str]:
        """执行工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        results = {}
        
        if workflow.mode == CoordinationMode.SEQUENTIAL:
            # 拓扑排序执行任务
            executed = set()
            pending = set(workflow.tasks.keys())
            
            while pending:
                # 找到可以执行的任务
                ready = []
                for task_id in pending:
                    task = workflow.tasks[task_id]
                    if all(dep in executed for dep in task.dependencies):
                        ready.append(task_id)
                
                if not ready:
                    raise Exception("Circular dependency detected")
                
                # 执行任务
                for task_id in ready:
                    task = workflow.tasks[task_id]
                    
                    if task.assigned_agent and task.assigned_agent in self.teams:
                        team = self.teams[task.assigned_agent]
                        result = await team.execute(task.description)
                    else:
                        # 如果没有指定团队，直接返回描述（模拟执行）
                        result = f"Executed: {task.description}"
                    
                    results[task_id] = result
                    task.status = "completed"
                    task.result = result
                    executed.add(task_id)
                    pending.remove(task_id)
        
        return results
    
    def get_team(self, team_id: str) -> Optional[AgentTeam]:
        """获取团队"""
        return self.teams.get(team_id)
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """获取工作流"""
        return self.workflows.get(workflow_id)
    
    def list_teams(self) -> List[str]:
        """列出所有团队"""
        return list(self.teams.keys())
    
    def list_workflows(self) -> List[str]:
        """列出所有工作流"""
        return list(self.workflows.keys())