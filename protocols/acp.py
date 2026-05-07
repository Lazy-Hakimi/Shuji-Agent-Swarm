"""
枢机 (Shuji) - Agent Communication Protocol (ACP)
实现基于WebSocket的ACP协议
"""
import json
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import uuid
import logging

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not installed. ACP client will be disabled.")


logger = logging.getLogger(__name__)


class ACPMessageType(Enum):
    """ACP消息类型"""
    INITIALIZE = "initialize"
    NEW_SESSION = "newSession"
    LOAD_SESSION = "loadSession"
    PROMPT = "prompt"
    CANCEL = "cancel"
    SESSION_UPDATE = "session/update"
    SESSION_INFO = "session/info"
    USAGE_UPDATE = "usage/update"
    TOOL_CALL = "tool/call"
    TOOL_CALL_UPDATE = "tool/callUpdate"
    ERROR = "error"


@dataclass
class ACPMessage:
    """ACP消息"""
    
    def __init__(
        self,
        type: str,
        id: Optional[str] = None,
        session_id: Optional[str] = None,
        payload: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        self.type = type
        self.id = id or str(uuid.uuid4())
        self.session_id = session_id
        self.payload = payload or {}
        self.error = error
    
    def to_dict(self) -> Dict:
        result = {"type": self.type, "id": self.id}
        if self.session_id:
            result["sessionId"] = self.session_id
        if self.payload:
            result["payload"] = self.payload
        if self.error:
            result["error"] = self.error
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ACPMessage':
        return cls(
            type=data.get("type", ""),
            id=data.get("id"),
            session_id=data.get("sessionId"),
            payload=data.get("payload"),
            error=data.get("error"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ACPMessage':
        return cls.from_dict(json.loads(json_str))


class ACPSession:
    """ACP会话"""
    
    def __init__(
        self,
        session_id: str,
        agent_id: str,
        workspace_dir: str,
        capabilities: Optional[Dict] = None,
    ):
        self.session_id = session_id
        self.agent_id = agent_id
        self.workspace_dir = workspace_dir
        self.capabilities = capabilities or {}
        self.messages: List[Dict] = []
        self.created_at = asyncio.get_event_loop().time()
        self.last_activity = self.created_at
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": asyncio.get_event_loop().time(),
        })
        self.last_activity = asyncio.get_event_loop().time()
    
    def to_dict(self) -> Dict:
        return {
            "sessionId": self.session_id,
            "agentId": self.agent_id,
            "workspaceDir": self.workspace_dir,
            "capabilities": self.capabilities,
            "messageCount": len(self.messages),
            "createdAt": self.created_at,
            "lastActivity": self.last_activity,
        }


class ACPServer:
    """
    ACP服务器
    """
    
    def __init__(self):
        self.sessions: Dict[str, ACPSession] = {}
        self.agent_sessions: Dict[str, List[str]] = {}  # agent_id -> session_ids
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler(self, message_type: str, handler: Callable):
        """注册处理器"""
        self.handlers[message_type] = handler
    
    async def handle_message(self, message: ACPMessage) -> ACPMessage:
        """处理消息"""
        if message.type in self.handlers:
            try:
                result = await self.handlers[message.type](message)
                return ACPMessage(
                    type=f"{message.type}/response",
                    id=message.id,
                    payload=result,
                )
            except Exception as e:
                logger.error(f"Handler error for {message.type}: {e}")
                return ACPMessage(
                    type="error",
                    id=message.id,
                    error=str(e),
                )
        else:
            return ACPMessage(
                type="error",
                id=message.id,
                error=f"Unknown message type: {message.type}",
            )
    
    def create_session(
        self,
        agent_id: str,
        workspace_dir: str,
        capabilities: Optional[Dict] = None,
    ) -> ACPSession:
        """创建会话"""
        session_id = str(uuid.uuid4())
        session = ACPSession(
            session_id=session_id,
            agent_id=agent_id,
            workspace_dir=workspace_dir,
            capabilities=capabilities,
        )
        
        self.sessions[session_id] = session
        
        if agent_id not in self.agent_sessions:
            self.agent_sessions[agent_id] = []
        self.agent_sessions[agent_id].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ACPSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """关闭会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # 从agent_sessions中移除
            if session.agent_id in self.agent_sessions:
                if session_id in self.agent_sessions[session.agent_id]:
                    self.agent_sessions[session.agent_id].remove(session_id)
            
            del self.sessions[session_id]
    
    def list_sessions(self, agent_id: Optional[str] = None) -> List[Dict]:
        """列出会话"""
        if agent_id:
            session_ids = self.agent_sessions.get(agent_id, [])
            return [
                self.sessions[sid].to_dict()
                for sid in session_ids
                if sid in self.sessions
            ]
        else:
            return [session.to_dict() for session in self.sessions.values()]
    
    async def send_to_session(self, session_id: str, message: ACPMessage):
        """发送消息到会话（需要外部实现WebSocket推送）"""
        # 实际应用中，需要与Gateway集成
        logger.info(f"Send to session {session_id}: {message.type}")
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.add_message("assistant", message.payload.get("content", ""))


class ACPClient:
    """ACP客户端（基于WebSocket）"""
    
    def __init__(self, server_url: str, agent_id: str):
        self.server_url = server_url
        self.agent_id = agent_id
        self.session_id: Optional[str] = None
        self.connected = False
        self.ws: Optional[WebSocketClientProtocol] = None
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._listener_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """连接服务器"""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required")
        
        try:
            self.ws = await websockets.connect(self.server_url)
            self.connected = True
            # 启动监听器
            self._listener_task = asyncio.create_task(self._listen())
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ACP server: {e}")
            return False
    
    async def _listen(self):
        """监听传入消息"""
        while self.connected and self.ws:
            try:
                message = await self.ws.recv()
                acp_msg = ACPMessage.from_json(message)
                
                # 如果是响应，完成对应的future
                if acp_msg.id in self._response_futures:
                    future = self._response_futures.pop(acp_msg.id)
                    if not future.done():
                        if acp_msg.error:
                            future.set_exception(Exception(acp_msg.error))
                        else:
                            future.set_result(acp_msg)
                else:
                    # 其他消息，可以交给上层处理
                    logger.debug(f"Received unsolicited message: {acp_msg.type}")
                    
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error in ACP listener: {e}")
                break
        
        self.connected = False
    
    async def _send(self, message: ACPMessage) -> ACPMessage:
        """发送消息并等待响应"""
        if not self.connected or not self.ws:
            raise Exception("Not connected")
        
        future = asyncio.get_event_loop().create_future()
        self._response_futures[message.id] = future
        
        try:
            await self.ws.send(message.to_json())
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self._response_futures.pop(message.id, None)
            raise Exception("Timeout waiting for response")
    
    async def create_session(
        self,
        workspace_dir: str,
        capabilities: Optional[Dict] = None,
    ) -> str:
        """创建会话"""
        message = ACPMessage(
            type=ACPMessageType.NEW_SESSION.value,
            payload={
                "agentId": self.agent_id,
                "workspaceDir": workspace_dir,
                "capabilities": capabilities or {},
            }
        )
        
        response = await self._send(message)
        self.session_id = response.payload.get("sessionId")
        return self.session_id
    
    async def send_prompt(self, content: str, attachments: Optional[List] = None) -> Dict:
        """发送提示"""
        if not self.session_id:
            raise Exception("No active session")
        
        message = ACPMessage(
            type=ACPMessageType.PROMPT.value,
            session_id=self.session_id,
            payload={
                "content": content,
                "attachments": attachments or [],
            }
        )
        
        response = await self._send(message)
        return response.payload
    
    async def cancel(self) -> Dict:
        """取消当前操作"""
        if not self.session_id:
            raise Exception("No active session")
        
        message = ACPMessage(
            type=ACPMessageType.CANCEL.value,
            session_id=self.session_id,
        )
        
        response = await self._send(message)
        return response.payload
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
        self.session_id = None
        if self.ws:
            await self.ws.close()
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass