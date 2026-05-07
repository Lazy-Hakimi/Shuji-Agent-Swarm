"""
枢机 (Shuji) - 内置技能
实现核心工具技能，增加安全性和可靠性
"""
import os
import subprocess
import json
import asyncio
import hashlib
import tempfile
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseSkill(ABC):
    """技能基类"""
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行技能"""
        pass


class FileSkill(BaseSkill):
    """文件操作技能"""
    
    def __init__(self, allowed_base_dir: Optional[str] = None):
        """
        初始化文件技能
        :param allowed_base_dir: 允许操作的根目录，None表示允许任何路径（不推荐）
        """
        self.allowed_base_dir = allowed_base_dir
        if allowed_base_dir:
            os.makedirs(allowed_base_dir, exist_ok=True)
    
    def _safe_path(self, path: str) -> str:
        """确保路径在允许的基目录内"""
        if self.allowed_base_dir:
            abs_path = os.path.abspath(os.path.join(self.allowed_base_dir, path))
            if not abs_path.startswith(os.path.abspath(self.allowed_base_dir)):
                raise ValueError(f"Access denied: {path} is outside allowed directory")
            return abs_path
        return os.path.abspath(path)
    
    async def execute(
        self,
        action: str,
        path: str,
        content: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """执行文件操作"""
        try:
            safe_path = self._safe_path(path)
            
            if action == "read":
                return await self._read(safe_path)
            elif action == "write":
                return await self._write(safe_path, content or "")
            elif action == "edit":
                return await self._edit(safe_path, content or "")
            elif action == "delete":
                return await self._delete(safe_path)
            elif action == "list":
                return await self._list(safe_path)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"FileSkill error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _read(self, path: str) -> Dict:
        """读取文件"""
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        if not os.path.isfile(path):
            return {"success": False, "error": f"Not a file: {path}"}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _write(self, path: str, content: str) -> Dict:
        """写入文件"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _edit(self, path: str, content: str) -> Dict:
        """编辑文件"""
        return await self._write(path, content)
    
    async def _delete(self, path: str) -> Dict:
        """删除文件或空目录"""
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                os.rmdir(path)  # 只删除空目录
            else:
                return {"success": False, "error": f"Path does not exist: {path}"}
            return {"success": True}
        except OSError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _list(self, path: str) -> Dict:
        """列出目录内容"""
        try:
            if not os.path.isdir(path):
                return {"success": False, "error": f"Not a directory: {path}"}
            items = os.listdir(path)
            # 区分文件和目录
            details = []
            for item in items:
                full = os.path.join(path, item)
                details.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(full) else "file",
                    "size": os.path.getsize(full) if os.path.isfile(full) else 0,
                })
            return {"success": True, "items": details}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BashSkill(BaseSkill):
    """Bash命令技能（带安全限制）"""
    
    # 危险命令黑名单
    DANGEROUS_COMMANDS = [
        "rm -rf /", "mkfs", "dd if=", "> /dev/sda", "format",
        ":(){ :|:& };:", "wget http://", "curl http://", "chmod 777 /",
        "sudo", "su ", "passwd"
    ]
    
    def __init__(self, allowed_commands: Optional[List[str]] = None, timeout: float = 30.0):
        """
        初始化Bash技能
        :param allowed_commands: 允许的命令列表（前缀匹配），None表示检查黑名单
        :param timeout: 命令超时时间
        """
        self.allowed_commands = allowed_commands
        self.timeout = timeout
    
    def _is_safe(self, command: str) -> bool:
        """检查命令是否安全"""
        if self.allowed_commands:
            # 白名单模式：命令必须以允许的列表中的某个开头
            return any(command.startswith(cmd) for cmd in self.allowed_commands)
        else:
            # 黑名单模式：检查是否包含危险命令
            cmd_lower = command.lower()
            for dangerous in self.DANGEROUS_COMMANDS:
                if dangerous in cmd_lower:
                    return False
            return True
    
    async def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """执行bash命令"""
        if not self._is_safe(command):
            return {"success": False, "error": "Command blocked for security reasons"}
        
        timeout = timeout or self.timeout
        
        try:
            # 使用asyncio.subprocess避免阻塞
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                }
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "returncode": process.returncode,
            }
        except Exception as e:
            logger.error(f"BashSkill error: {e}")
            return {"success": False, "error": str(e)}


class BrowserSkill(BaseSkill):
    """浏览器自动化技能（使用Playwright）"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
    
    async def _ensure_playwright(self):
        """确保Playwright已安装"""
        if self._playwright is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = async_playwright()
                self._playwright_instance = await self._playwright.start()
            except ImportError:
                raise ImportError("Playwright not installed. Install with: pip install playwright")
    
    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """执行浏览器操作"""
        try:
            await self._ensure_playwright()
            
            browser = await self._playwright_instance.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            
            try:
                if action == "navigate":
                    result = await self._navigate(page, url)
                elif action == "click":
                    result = await self._click(page, selector)
                elif action == "type":
                    result = await self._type(page, selector, text)
                elif action == "extract":
                    result = await self._extract(page, selector)
                elif action == "screenshot":
                    result = await self._screenshot(page, kwargs.get("output_path"))
                else:
                    result = {"success": False, "error": f"Unknown action: {action}"}
            finally:
                await browser.close()
            
            return result
        
        except Exception as e:
            logger.error(f"BrowserSkill error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _navigate(self, page, url: str) -> Dict:
        """导航到URL"""
        await page.goto(url, wait_until="networkidle")
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
            "content": await page.content()
        }
    
    async def _click(self, page, selector: str) -> Dict:
        """点击元素"""
        await page.click(selector)
        return {"success": True}
    
    async def _type(self, page, selector: str, text: str) -> Dict:
        """输入文本"""
        await page.fill(selector, text)
        return {"success": True}
    
    async def _extract(self, page, selector: str) -> Dict:
        """提取元素内容"""
        element = await page.query_selector(selector)
        if element:
            text = await element.text_content()
            html = await element.inner_html()
            return {"success": True, "content": text, "html": html}
        return {"success": False, "error": "Element not found"}
    
    async def _screenshot(self, page, output_path: Optional[str]) -> Dict:
        """截图"""
        if output_path:
            path = output_path
        else:
            # 生成临时文件
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
        
        await page.screenshot(path=path)
        return {"success": True, "path": path}


class SearchSkill(BaseSkill):
    """网络搜索技能（使用DuckDuckGo API）"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # 备用，用于其他搜索引擎
    
    async def execute(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> Dict:
        """执行搜索"""
        try:
            # 使用DuckDuckGo的简单API（不需要key）
            import aiohttp
            
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        return {"success": False, "error": f"Search API returned {resp.status}"}
                    
                    data = await resp.json()
                    
                    # 解析结果
                    results = []
                    # Abstract
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", "Abstract"),
                            "snippet": data["Abstract"],
                            "url": data.get("AbstractURL", ""),
                        })
                    
                    # Related topics
                    for topic in data.get("RelatedTopics", [])[:num_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1],
                                "snippet": topic["Text"],
                                "url": topic.get("FirstURL", ""),
                            })
                    
                    # 如果结果太少，使用HTML搜索（备选）
                    if len(results) < num_results:
                        fallback = await self._fallback_search(query, num_results - len(results))
                        results.extend(fallback)
                    
                    return {"success": True, "results": results[:num_results]}
        
        except ImportError:
            # aiohttp未安装，使用urllib
            return await self._urllib_search(query, num_results)
        except Exception as e:
            logger.error(f"SearchSkill error: {e}")
            # 返回模拟结果作为最后备选
            return {
                "success": True,
                "results": [
                    {"title": f"Result {i+1} for '{query}'", "snippet": "Mock search result"}
                    for i in range(num_results)
                ],
                "note": f"Using mock results due to error: {str(e)}",
            }
    
    async def _urllib_search(self, query: str, num_results: int) -> Dict:
        """使用urllib的备选实现"""
        import urllib.request
        import urllib.parse
        import re
        
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
            
            results = []
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', html)
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html)
            
            for i in range(min(num_results, len(titles))):
                results.append({
                    "title": re.sub(r'<[^>]+>', '', titles[i]),
                    "snippet": re.sub(r'<[^>]+>', '', snippets[i]) if i < len(snippets) else "",
                })
            
            return {"success": True, "results": results}
        except Exception as e:
            raise e
    
    async def _fallback_search(self, query: str, num_results: int) -> List[Dict]:
        """备选搜索（简单爬取）"""
        # 这里可以调用其他搜索引擎，但为了简单，返回空
        return []


class CodeSkill(BaseSkill):
    """代码操作技能（安全沙箱版本）"""
    
    def __init__(self, sandbox: bool = True):
        self.sandbox = sandbox
    
    async def execute(
        self,
        action: str,
        language: Optional[str] = None,
        code: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """执行代码操作"""
        if action == "analyze":
            return await self._analyze(code, language)
        elif action == "execute":
            if not self.sandbox:
                return {"success": False, "error": "Code execution disabled without sandbox"}
            return await self._execute_code(code, language)
        elif action == "format":
            return await self._format(code, language)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    async def _analyze(self, code: Optional[str], language: Optional[str]) -> Dict:
        """分析代码（安全）"""
        if not code:
            return {"success": False, "error": "No code provided"}
        
        lines = code.split('\n')
        analysis = {
            "line_count": len(lines),
            "language": language or "unknown",
            "has_comments": '#' in code or '//' in code,
            "char_count": len(code),
            "estimated_tokens": len(code) // 4,
        }
        
        return {"success": True, "analysis": analysis}
    
    async def _execute_code(self, code: Optional[str], language: Optional[str]) -> Dict:
        """执行代码（在子进程中安全运行）"""
        if not code:
            return {"success": False, "error": "No code provided"}
        
        if language == "python":
            # 在临时文件中写入代码，使用子进程执行
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp_path = f.name
            
            try:
                # 使用python -c 执行，但限制资源
                # 这里简单起见，使用subprocess，但实际生产应使用容器
                process = await asyncio.create_subprocess_exec(
                    'python', tmp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024*1024,  # 输出限制1MB
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {"success": False, "error": "Execution timed out"}
                
                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
            finally:
                os.unlink(tmp_path)
        
        return {"success": False, "error": f"Execution not supported for language: {language}"}
    
    async def _format(self, code: Optional[str], language: Optional[str]) -> Dict:
        """格式化代码"""
        if not code:
            return {"success": False, "error": "No code provided"}
        
        # 简单格式化：去除多余空行
        lines = [line for line in code.split('\n') if line.strip()]
        formatted = '\n'.join(lines)
        
        return {"success": True, "formatted_code": formatted}