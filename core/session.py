"""
枢机 (Shuji) - 会话管理
管理用户会话和上下文
"""
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid


@dataclass
class Session:
    """会话定义"""
    session_id: str
    user_id: str
    agent_id: str
    channel: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """更新活动时间"""
        self.last_activity = time.time()
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "channel": self.channel,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "context_keys": list(self.context.keys()),
        }


class SessionManager:
    """
    会话管理器
    
    管理所有活跃的会话
    """
    
    def __init__(self, max_sessions: int = 1000, session_timeout: float = 3600.0):
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self.agent_sessions: Dict[str, List[str]] = {}  # agent_id -> session_ids
        
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
    
    def create_session(
        self,
        user_id: str,
        agent_id: str,
        channel: str = "cli",
        context: Optional[Dict] = None,
    ) -> Session:
        """创建会话"""
        session_id = str(uuid.uuid4())
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            channel=channel,
            context=context or {},
        )
        
        self.sessions[session_id] = session
        
        # 更新索引
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        if agent_id not in self.agent_sessions:
            self.agent_sessions[agent_id] = []
        self.agent_sessions[agent_id].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session
    
    def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # 从索引中移除
        if session.user_id in self.user_sessions:
            if session_id in self.user_sessions[session.user_id]:
                self.user_sessions[session.user_id].remove(session_id)
        
        if session.agent_id in self.agent_sessions:
            if session_id in self.agent_sessions[session.agent_id]:
                self.agent_sessions[session.agent_id].remove(session_id)
        
        del self.sessions[session_id]
        return True
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """获取用户的所有会话"""
        session_ids = self.user_sessions.get(user_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def get_agent_sessions(self, agent_id: str) -> List[Session]:
        """获取Agent的所有会话"""
        session_ids = self.agent_sessions.get(agent_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict]:
        """列出会话"""
        if user_id:
            sessions = self.get_user_sessions(user_id)
        elif agent_id:
            sessions = self.get_agent_sessions(agent_id)
        else:
            sessions = list(self.sessions.values())
        
        return [session.to_dict() for session in sessions]
    
    def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        now = time.time()
        expired = []
        
        for session_id, session in self.sessions.items():
            if now - session.last_activity > self.session_timeout:
                expired.append(session_id)
        
        for session_id in expired:
            self.close_session(session_id)
        
        return len(expired)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_sessions": len(self.sessions),
            "unique_users": len(self.user_sessions),
            "unique_agents": len(self.agent_sessions),
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout,
        }
    
    # Gateway handlers
    async def handle_create(self, connection, params: Dict) -> Dict:
        """处理创建会话请求"""
        user_id = params.get("user_id", "anonymous")
        agent_id = params.get("agent_id")
        channel = params.get("channel", "cli")
        context = params.get("context", {})
        
        session = self.create_session(user_id, agent_id, channel, context)
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
        }
    
    async def handle_load(self, connection, params: Dict) -> Dict:
        """处理加载会话请求"""
        session_id = params.get("session_id")
        
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        return session.to_dict()
    
    async def handle_list(self, connection, params: Dict) -> Dict:
        """处理列会话请求"""
        user_id = params.get("user_id")
        agent_id = params.get("agent_id")
        
        sessions = self.list_sessions(user_id, agent_id)
        return {"sessions": sessions}
    
    async def handle_close(self, connection, params: Dict) -> Dict:
        """处理关闭会话请求"""
        session_id = params.get("session_id")
        
        success = self.close_session(session_id)
        return {"success": success, "session_id": session_id}