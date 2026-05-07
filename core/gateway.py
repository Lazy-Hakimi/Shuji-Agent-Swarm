"""
枢机 (Shuji) - Gateway网关系统
实现OpenClaw风格的WebSocket网关
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class MessageType(Enum):
    """消息类型"""
    REQUEST = "req"
    RESPONSE = "res"
    EVENT = "event"
    HELLO = "hello"
    PING = "ping"
    PONG = "pong"


class EventType(Enum):
    """事件类型"""
    AGENT_OUTPUT = "agent"
    CHAT_MESSAGE = "chat"
    PRESENCE = "presence"
    HEALTH = "health"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class GatewayMessage:
    """网关消息格式"""
    type: str
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict] = None
    payload: Optional[Dict] = None
    ok: Optional[bool] = None
    event: Optional[str] = None
    seq: Optional[int] = None
    state_version: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {"type": self.type}
        if self.id is not None:
            result["id"] = self.id
        if self.method is not None:
            result["method"] = self.method
        if self.params is not None:
            result["params"] = self.params
        if self.payload is not None:
            result["payload"] = self.payload
        if self.ok is not None:
            result["ok"] = self.ok
        if self.event is not None:
            result["event"] = self.event
        if self.seq is not None:
            result["seq"] = self.seq
        if self.state_version is not None:
            result["stateVersion"] = self.state_version
        if self.error is not None:
            result["error"] = self.error
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GatewayMessage':
        """从字典创建消息"""
        return cls(
            type=data.get("type", ""),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            payload=data.get("payload"),
            ok=data.get("ok"),
            event=data.get("event"),
            seq=data.get("seq"),
            state_version=data.get("stateVersion"),
            error=data.get("error"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GatewayMessage':
        """从JSON字符串创建消息"""
        return cls.from_dict(json.loads(json_str))


class Connection:
    """WebSocket连接封装"""
    
    def __init__(self, websocket: WebSocketServerProtocol, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = time.time()
        self.last_activity = time.time()
        self.authenticated = False
        self.role: Optional[str] = None
        self.agent_id: Optional[str] = None
        self.session_key: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    async def send(self, message: GatewayMessage):
        """发送消息"""
        await self.websocket.send(message.to_json())
        self.last_activity = time.time()
    
    async def close(self):
        """关闭连接"""
        await self.websocket.close()


class ShujiGateway:
    """
    枢机网关系统
    
    对应OpenClaw Gateway的核心功能:
    - WebSocket服务器
    - 消息路由
    - 连接管理
    - 事件广播
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("ShujiGateway")
        
        # 连接管理
        self.connections: Dict[str, Connection] = {}
        self.agent_connections: Dict[str, str] = {}  # agent_id -> connection_id
        
        # 会话管理
        self.sessions: Dict[str, Dict] = {}
        
        # 消息处理器
        self.handlers: Dict[str, Callable] = {}
        
        # 事件订阅者
        self.event_subscribers: Dict[str, List[str]] = {}
        
        # 统计信息
        self.stats = {
            "total_connections": 0,
            "total_messages": 0,
            "start_time": None,
        }
        
        # 运行状态
        self.running = False
        self.server = None
    
    def register_handler(self, method: str, handler: Callable):
        """注册消息处理器"""
        self.handlers[method] = handler
        self.logger.debug(f"Registered handler for method: {method}")
    
    def unregister_handler(self, method: str):
        """注销消息处理器"""
        if method in self.handlers:
            del self.handlers[method]
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """处理新连接"""
        connection_id = str(uuid.uuid4())
        connection = Connection(websocket, connection_id)
        self.connections[connection_id] = connection
        self.stats["total_connections"] += 1
        
        self.logger.info(f"New connection: {connection_id} from {websocket.remote_address}")
        
        try:
            # 发送握手响应
            await self._send_hello(connection)
            
            # 处理消息
            async for message in websocket:
                await self._handle_message(connection, message)
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"Error handling connection {connection_id}: {e}")
        finally:
            await self._cleanup_connection(connection)
    
    async def _send_hello(self, connection: Connection):
        """发送握手响应"""
        hello_message = GatewayMessage(
            type=MessageType.HELLO.value,
            payload={
                "protocolVersion": self.config.protocol_version,
                "gatewayVersion": "1.0.0",
                "serverTime": time.time(),
                "capabilities": ["agent", "chat", "session"],
            }
        )
        await connection.send(hello_message)
    
    async def _handle_message(self, connection: Connection, message: str):
        """处理消息"""
        try:
            gateway_msg = GatewayMessage.from_json(message)
            self.stats["total_messages"] += 1
            connection.last_activity = time.time()
            
            if gateway_msg.type == MessageType.PING.value:
                await self._handle_ping(connection)
            elif gateway_msg.type == MessageType.REQUEST.value:
                await self._handle_request(connection, gateway_msg)
            elif gateway_msg.type == MessageType.EVENT.value:
                await self._handle_event(connection, gateway_msg)
            else:
                await self._send_error(connection, gateway_msg.id, f"Unknown message type: {gateway_msg.type}")
        
        except json.JSONDecodeError:
            await self._send_error(connection, None, "Invalid JSON")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            await self._send_error(connection, None, str(e))
    
    async def _handle_ping(self, connection: Connection):
        """处理ping消息"""
        pong = GatewayMessage(
            type=MessageType.PONG.value,
            payload={"time": time.time()}
        )
        await connection.send(pong)
    
    async def _handle_request(self, connection: Connection, message: GatewayMessage):
        """处理请求"""
        method = message.method
        params = message.params or {}
        
        if method in self.handlers:
            try:
                result = await self.handlers[method](connection, params)
                response = GatewayMessage(
                    type=MessageType.RESPONSE.value,
                    id=message.id,
                    ok=True,
                    payload=result
                )
            except Exception as e:
                self.logger.error(f"Handler error for {method}: {e}")
                response = GatewayMessage(
                    type=MessageType.RESPONSE.value,
                    id=message.id,
                    ok=False,
                    error=str(e)
                )
        else:
            response = GatewayMessage(
                type=MessageType.RESPONSE.value,
                id=message.id,
                ok=False,
                error=f"Unknown method: {method}"
            )
        
        await connection.send(response)
    
    async def _handle_event(self, connection: Connection, message: GatewayMessage):
        """处理事件"""
        event_type = message.event
        # 广播事件给订阅者
        if event_type in self.event_subscribers:
            await self.broadcast_event(event_type, message.payload)
    
    async def _send_error(self, connection: Connection, request_id: Optional[str], error: str):
        """发送错误响应"""
        response = GatewayMessage(
            type=MessageType.RESPONSE.value,
            id=request_id,
            ok=False,
            error=error
        )
        await connection.send(response)
    
    async def _cleanup_connection(self, connection: Connection):
        """清理连接"""
        if connection.agent_id and connection.agent_id in self.agent_connections:
            del self.agent_connections[connection.agent_id]
        
        if connection.connection_id in self.connections:
            del self.connections[connection.connection_id]
        
        self.logger.info(f"Cleaned up connection: {connection.connection_id}")
    
    async def broadcast_event(self, event_type: str, payload: Dict, exclude: Optional[List[str]] = None):
        """广播事件"""
        exclude = exclude or []
        event = GatewayMessage(
            type=MessageType.EVENT.value,
            event=event_type,
            payload=payload,
            seq=int(time.time() * 1000)
        )
        
        tasks = []
        for conn_id, conn in self.connections.items():
            if conn_id not in exclude:
                tasks.append(conn.send(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_to_agent(self, agent_id: str, event_type: str, payload: Dict):
        """发送事件给特定Agent"""
        if agent_id in self.agent_connections:
            conn_id = self.agent_connections[agent_id]
            if conn_id in self.connections:
                event = GatewayMessage(
                    type=MessageType.EVENT.value,
                    event=event_type,
                    payload=payload
                )
                await self.connections[conn_id].send(event)
    
    async def start(self):
        """启动网关"""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required. Install with: pip install websockets")
        
        self.running = True
        self.stats["start_time"] = time.time()
        
        self.server = await websockets.serve(
            self.handle_connection,
            self.config.gateway_host,
            self.config.gateway_port,
            ping_interval=self.config.heartbeat_interval,
            ping_timeout=self.config.connection_timeout,
        )
        
        self.logger.info(f"Gateway started on {self.config.gateway_host}:{self.config.gateway_port}")
        
        # 启动心跳检查
        asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self):
        """停止网关"""
        self.running = False
        
        # 关闭所有连接
        for conn in list(self.connections.values()):
            await conn.close()
        
        # 关闭服务器
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.logger.info("Gateway stopped")
    
    async def _heartbeat_loop(self):
        """心跳检查循环"""
        while self.running:
            await asyncio.sleep(self.config.heartbeat_interval)
            
            # 检查不活跃的连接
            now = time.time()
            timeout = self.config.connection_timeout
            
            stale_connections = [
                conn for conn in self.connections.values()
                if now - conn.last_activity > timeout
            ]
            
            for conn in stale_connections:
                self.logger.warning(f"Closing stale connection: {conn.connection_id}")
                await conn.close()
    
    def get_health(self) -> Dict:
        """获取健康状态"""
        return {
            "status": "healthy" if self.running else "stopped",
            "connections": len(self.connections),
            "agents": len(self.agent_connections),
            "sessions": len(self.sessions),
            "uptime": time.time() - self.stats["start_time"] if self.stats["start_time"] else 0,
            "total_connections": self.stats["total_connections"],
            "total_messages": self.stats["total_messages"],
        }


class GatewayServer:
    """网关服务器封装"""
    
    def __init__(self, config):
        self.config = config
        self.gateway = ShujiGateway(config)
        self.logger = logging.getLogger("GatewayServer")
    
    def register_agent_handler(self, agent_manager):
        """注册Agent处理器"""
        # 注册agent相关的方法
        self.gateway.register_handler("agent.chat", agent_manager.handle_chat)
        self.gateway.register_handler("agent.run", agent_manager.handle_run)
        self.gateway.register_handler("agent.status", agent_manager.handle_status)
        self.gateway.register_handler("agent.list", agent_manager.handle_list)
        self.gateway.register_handler("agent.create", agent_manager.handle_create)
        self.gateway.register_handler("agent.delete", agent_manager.handle_delete)
    
    def register_session_handler(self, session_manager):
        """注册Session处理器"""
        self.gateway.register_handler("session.create", session_manager.handle_create)
        self.gateway.register_handler("session.load", session_manager.handle_load)
        self.gateway.register_handler("session.list", session_manager.handle_list)
        self.gateway.register_handler("session.close", session_manager.handle_close)
    
    async def start(self):
        """启动服务器"""
        await self.gateway.start()
    
    async def stop(self):
        """停止服务器"""
        await self.gateway.stop()
    
    async def run_forever(self):
        """持续运行"""
        await self.start()
        try:
            while self.gateway.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop()