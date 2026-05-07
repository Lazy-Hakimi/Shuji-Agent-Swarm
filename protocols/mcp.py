"""
枢机 (Shuji) - Model Context Protocol (MCP)
实现基于WebSocket的MCP协议（JSON-RPC 2.0）
"""
import json
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import logging

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not installed. MCP client will be disabled.")


logger = logging.getLogger(__name__)


class MCPMethod(Enum):
    """MCP方法"""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"


@dataclass
class MCPTool:
    """MCP工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """MCP资源定义"""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    
    def to_dict(self) -> Dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPrompt:
    """MCP提示定义"""
    name: str
    description: str
    arguments: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        result = {
            "name": self.name,
            "description": self.description,
        }
        if self.arguments:
            result["arguments"] = self.arguments
        return result


class MCPMessage:
    """MCP消息（JSON-RPC 2.0）"""
    
    def __init__(
        self,
        jsonrpc: str = "2.0",
        id: Optional[str] = None,
        method: Optional[str] = None,
        params: Optional[Dict] = None,
        result: Optional[Dict] = None,
        error: Optional[Dict] = None,
    ):
        self.jsonrpc = jsonrpc
        self.id = id or str(uuid.uuid4())
        self.method = method
        self.params = params or {}
        self.result = result
        self.error = error
    
    def to_dict(self) -> Dict:
        result = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.method:
            result["method"] = self.method
            result["params"] = self.params
        if self.result is not None:
            result["result"] = self.result
        if self.error is not None:
            result["error"] = self.error
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPMessage':
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def create_request(cls, method: str, params: Dict) -> 'MCPMessage':
        """创建请求消息"""
        return cls(method=method, params=params)
    
    @classmethod
    def create_response(cls, request_id: str, result: Dict) -> 'MCPMessage':
        """创建响应消息"""
        return cls(id=request_id, result=result)
    
    @classmethod
    def create_error(cls, request_id: str, code: int, message: str) -> 'MCPMessage':
        """创建错误消息"""
        return cls(
            id=request_id,
            error={"code": code, "message": message}
        )


class MCPServer:
    """
    MCP服务器
    """
    
    def __init__(self, name: str = "shuji-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        
        # 注册表
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        
        # 处理器
        self.handlers: Dict[str, Callable] = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.RESOURCES_LIST.value: self._handle_resources_list,
            MCPMethod.RESOURCES_READ.value: self._handle_resources_read,
            MCPMethod.PROMPTS_LIST.value: self._handle_prompts_list,
            MCPMethod.PROMPTS_GET.value: self._handle_prompts_get,
        }
    
    def register_tool(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def unregister_tool(self, name: str):
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
    
    def register_resource(self, resource: MCPResource):
        """注册资源"""
        self.resources[resource.uri] = resource
    
    def register_prompt(self, prompt: MCPPrompt):
        """注册提示"""
        self.prompts[prompt.name] = prompt
    
    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """处理消息"""
        if message.method in self.handlers:
            try:
                result = await self.handlers[message.method](message.params)
                return MCPMessage.create_response(message.id, result)
            except Exception as e:
                logger.error(f"Handler error for {message.method}: {e}")
                return MCPMessage.create_error(message.id, -32603, str(e))
        else:
            return MCPMessage.create_error(
                message.id, -32601, f"Method not found: {message.method}"
            )
    
    async def _handle_initialize(self, params: Dict) -> Dict:
        """处理初始化"""
        client_info = params.get("clientInfo", {})
        protocol_version = params.get("protocolVersion", "2024-11-05")
        
        return {
            "protocolVersion": protocol_version,
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True, "listChanged": True},
                "prompts": {"listChanged": True},
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }
    
    async def _handle_tools_list(self, params: Dict) -> Dict:
        """处理工具列表"""
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()]
        }
    
    async def _handle_tools_call(self, params: Dict) -> Dict:
        """处理工具调用"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        if tool.handler is None:
            raise ValueError(f"Tool has no handler: {tool_name}")
        
        result = await tool.handler(**arguments)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": str(result),
                }
            ]
        }
    
    async def _handle_resources_list(self, params: Dict) -> Dict:
        """处理资源列表"""
        return {
            "resources": [resource.to_dict() for resource in self.resources.values()]
        }
    
    async def _handle_resources_read(self, params: Dict) -> Dict:
        """处理资源读取（实际从文件或数据库读取）"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
        
        resource = self.resources[uri]
        
        # 尝试读取实际内容
        content = f"Content of {uri}"  # 默认
        if uri.startswith("file://"):
            file_path = uri[7:]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                content = f"Error reading file: {e}"
        elif uri.startswith("memory://"):
            # 假设是记忆系统的URI
            # 可以集成记忆系统
            content = f"Memory content for {uri}"
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": content,
                }
            ]
        }
    
    async def _handle_prompts_list(self, params: Dict) -> Dict:
        """处理提示列表"""
        return {
            "prompts": [prompt.to_dict() for prompt in self.prompts.values()]
        }
    
    async def _handle_prompts_get(self, params: Dict) -> Dict:
        """处理提示获取"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name not in self.prompts:
            raise ValueError(f"Prompt not found: {name}")
        
        prompt = self.prompts[name]
        
        # 构建提示消息
        messages = [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"Prompt: {name} with args: {arguments}",
                }
            }
        ]
        
        return {
            "description": prompt.description,
            "messages": messages,
        }


class MCPClient:
    """MCP客户端（基于WebSocket）"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.initialized = False
        self.server_capabilities = {}
        self.ws: Optional[WebSocketClientProtocol] = None
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """连接服务器"""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required")
        
        try:
            self.ws = await websockets.connect(self.server_url)
            self.connected = True
            self._listener_task = asyncio.create_task(self._listen())
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def _listen(self):
        """监听传入消息"""
        while self.connected and self.ws:
            try:
                message = await self.ws.recv()
                mcp_msg = MCPMessage.from_json(message)
                
                if mcp_msg.id in self._response_futures:
                    future = self._response_futures.pop(mcp_msg.id)
                    if not future.done():
                        if mcp_msg.error:
                            future.set_exception(Exception(mcp_msg.error))
                        else:
                            future.set_result(mcp_msg)
                else:
                    logger.debug(f"Received unsolicited MCP message: {mcp_msg.id}")
                    
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error in MCP listener: {e}")
                break
        
        self.connected = False
    
    async def _send_request(self, request: MCPMessage) -> MCPMessage:
        """发送请求并等待响应"""
        if not self.connected or not self.ws:
            raise Exception("Not connected")
        
        future = asyncio.get_event_loop().create_future()
        self._response_futures[request.id] = future
        
        try:
            await self.ws.send(request.to_json())
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self._response_futures.pop(request.id, None)
            raise Exception("Timeout waiting for response")
    
    async def initialize(self) -> Dict:
        """初始化连接"""
        request = MCPMessage.create_request(
            MCPMethod.INITIALIZE.value,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "shuji-mcp-client",
                    "version": "1.0.0",
                },
            }
        )
        
        response = await self._send_request(request)
        
        if response.error:
            raise Exception(f"Initialization failed: {response.error}")
        
        self.server_capabilities = response.result.get("capabilities", {})
        self.initialized = True
        
        return response.result
    
    async def list_tools(self) -> List[Dict]:
        """列出可用工具"""
        request = MCPMessage.create_request(
            MCPMethod.TOOLS_LIST.value,
            {}
        )
        
        response = await self._send_request(request)
        return response.result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict) -> Dict:
        """调用工具"""
        request = MCPMessage.create_request(
            MCPMethod.TOOLS_CALL.value,
            {"name": name, "arguments": arguments}
        )
        
        response = await self._send_request(request)
        return response.result
    
    async def list_resources(self) -> List[Dict]:
        """列出可用资源"""
        request = MCPMessage.create_request(
            MCPMethod.RESOURCES_LIST.value,
            {}
        )
        
        response = await self._send_request(request)
        return response.result.get("resources", [])
    
    async def read_resource(self, uri: str) -> Dict:
        """读取资源"""
        request = MCPMessage.create_request(
            MCPMethod.RESOURCES_READ.value,
            {"uri": uri}
        )
        
        response = await self._send_request(request)
        return response.result
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
        if self.ws:
            await self.ws.close()
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass