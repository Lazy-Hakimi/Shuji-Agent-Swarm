"""
枢机 (Shuji) - 三层记忆系统
实现OpenClaw风格的三层记忆架构
"""
import os
import json
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MemoryLayer(Enum):
    """记忆层级"""
    L1_MARKDOWN = "markdown"
    L2_VECTOR = "vector"
    L3_GRAPH = "graph"


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    timestamp: str
    source: str
    memory_type: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "type": self.memory_type,
            "metadata": self.metadata,
        }


class MemorySystem:
    """
    三层记忆系统
    """
    
    def __init__(self, config):
        self.config = config
        self.workspace_dir = config.workspace_dir
        self.memory_dir = os.path.join(self.workspace_dir, "memory")
        
        # 确保目录存在
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 三层存储
        self.markdown_store = MarkdownStore(self.memory_dir)
        self.vector_store = None  # 延迟初始化
        self.knowledge_graph = None  # 延迟初始化
        
        # 索引状态：memory_id -> content_hash
        self.indexed_hashes: Dict[str, str] = {}
        self._load_index_state()
    
    def _load_index_state(self):
        """加载索引状态"""
        state_file = os.path.join(self.memory_dir, "index_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    self.indexed_hashes = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load index state: {e}")
    
    def _save_index_state(self):
        """保存索引状态"""
        state_file = os.path.join(self.memory_dir, "index_state.json")
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.indexed_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index state: {e}")
    
    async def initialize(self):
        """初始化向量存储和知识图谱"""
        # 初始化向量存储
        try:
            from .vector_store import ChromaVectorStore
            self.vector_store = ChromaVectorStore(
                collection_name="shuji_memory",
                persist_directory=os.path.join(self.memory_dir, "chroma")
            )
        except ImportError:
            logger.warning("ChromaDB not available, vector search disabled")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
        
        # 初始化知识图谱
        try:
            from .knowledge_graph import NetworkXKnowledgeGraph
            self.knowledge_graph = NetworkXKnowledgeGraph(
                graph_path=os.path.join(self.memory_dir, "knowledge_graph.json")
            )
        except ImportError:
            logger.warning("NetworkX not available, knowledge graph disabled")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
    
    async def add(
        self,
        content: str,
        memory_type: str = "conversation",
        source: str = "user",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加记忆
        """
        timestamp = datetime.now().isoformat()
        memory_id = hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()
        
        entry = MemoryEntry(
            content=content,
            timestamp=timestamp,
            source=source,
            memory_type=memory_type,
            metadata=metadata or {}
        )
        
        # L1: 保存到Markdown
        await self.markdown_store.add(entry)
        
        # L2: 添加到向量存储
        if self.vector_store:
            try:
                embedding = await self.vector_store.add(
                    id=memory_id,
                    text=content,
                    metadata=entry.to_dict()
                )
                entry.embedding = embedding
            except Exception as e:
                logger.error(f"Failed to add to vector store: {e}")
        
        # L3: 添加到知识图谱
        if self.knowledge_graph:
            try:
                await self.knowledge_graph.add_node(
                    node_id=memory_id,
                    content=content,
                    node_type=memory_type,
                    metadata=entry.to_dict()
                )
            except Exception as e:
                logger.error(f"Failed to add to knowledge graph: {e}")
        
        # 更新索引状态
        content_hash = hashlib.md5(content.encode()).hexdigest()
        self.indexed_hashes[memory_id] = content_hash
        self._save_index_state()
        
        return memory_id
    
    async def search(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[str]:
        """
        搜索记忆
        """
        results = []
        
        # L2: 向量搜索
        if self.vector_store:
            try:
                vector_results = await self.vector_store.search(
                    query=query,
                    k=k,
                    filter_dict={"type": memory_type} if memory_type else None
                )
                results.extend([r["content"] for r in vector_results])
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # L1: Markdown搜索（如果向量搜索不足）
        if len(results) < k:
            try:
                markdown_results = await self.markdown_store.search(
                    query=query,
                    k=k - len(results)
                )
                results.extend(markdown_results)
            except Exception as e:
                logger.error(f"Markdown search failed: {e}")
        
        # L3: 知识图谱相关内容
        if len(results) < k and self.knowledge_graph:
            try:
                graph_results = await self.knowledge_graph.get_related(query, k=k - len(results))
                results.extend(graph_results)
            except Exception as e:
                logger.error(f"Knowledge graph search failed: {e}")
        
        return results[:k]
    
    async def get_related(self, content: str, k: int = 5) -> List[str]:
        """获取相关内容 (知识图谱遍历)"""
        if self.knowledge_graph:
            try:
                return await self.knowledge_graph.get_related(content, k)
            except Exception as e:
                logger.error(f"Knowledge graph get_related failed: {e}")
        return []
    
    async def get_recent(self, n: int = 10, memory_type: Optional[str] = None) -> List[str]:
        """获取最近的记忆"""
        try:
            return await self.markdown_store.get_recent(n, memory_type)
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []
    
    async def sync(self):
        """同步三层记忆：检查新Markdown文件并索引"""
        try:
            new_entries = await self.markdown_store.get_unindexed(self.indexed_hashes)
            
            for entry in new_entries:
                memory_id = hashlib.md5(f"{entry.content}{entry.timestamp}".encode()).hexdigest()
                
                # 索引到向量存储
                if self.vector_store:
                    await self.vector_store.add(
                        id=memory_id,
                        text=entry.content,
                        metadata=entry.to_dict()
                    )
                
                # 添加到知识图谱
                if self.knowledge_graph:
                    await self.knowledge_graph.add_node(
                        node_id=memory_id,
                        content=entry.content,
                        node_type=entry.memory_type,
                        metadata=entry.to_dict()
                    )
                
                # 更新索引状态
                content_hash = hashlib.md5(entry.content.encode()).hexdigest()
                self.indexed_hashes[memory_id] = content_hash
            
            self._save_index_state()
            return len(new_entries)
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "indexed_entries": len(self.indexed_hashes),
            "vector_store": self.vector_store.get_stats() if self.vector_store else None,
            "knowledge_graph": self.knowledge_graph.get_stats() if self.knowledge_graph else None,
        }


class MarkdownStore:
    """Markdown文件存储 (L1)"""
    
    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        self.daily_dir = os.path.join(memory_dir, "daily")
        os.makedirs(self.daily_dir, exist_ok=True)
        
        self.memory_file = os.path.join(memory_dir, "MEMORY.md")
    
    async def add(self, entry: MemoryEntry):
        """添加记忆"""
        # 添加到每日日志
        date_str = entry.timestamp[:10]  # YYYY-MM-DD
        daily_file = os.path.join(self.daily_dir, f"{date_str}.md")
        
        with open(daily_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## {entry.timestamp}\n\n")
            f.write(f"**Source**: {entry.source}  \n")
            f.write(f"**Type**: {entry.memory_type}\n\n")
            f.write(f"{entry.content}\n")
        
        # 如果是重要记忆，也添加到MEMORY.md
        if entry.memory_type in ["important", "decision", "fact"]:
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(f"\n## {entry.timestamp} - {entry.memory_type.upper()}\n\n")
                f.write(f"{entry.content}\n")
    
    async def search(self, query: str, k: int = 5) -> List[str]:
        """搜索记忆（简单的全文搜索）"""
        results = []
        
        # 搜索所有Markdown文件
        for root, _, files in os.walk(self.memory_dir):
            for file in files:
                if file.endswith('.md'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if query.lower() in content.lower():
                                # 提取相关段落（简单：按双换行分割）
                                paragraphs = content.split('\n\n')
                                for para in paragraphs:
                                    if query.lower() in para.lower():
                                        results.append(para.strip())
                                        if len(results) >= k:
                                            return results
                    except Exception as e:
                        logger.error(f"Error reading {filepath}: {e}")
        
        return results
    
    async def get_recent(self, n: int = 10, memory_type: Optional[str] = None) -> List[str]:
        """获取最近的记忆"""
        # 获取所有每日日志文件
        try:
            daily_files = sorted(
                [f for f in os.listdir(self.daily_dir) if f.endswith('.md')],
                reverse=True
            )
        except FileNotFoundError:
            return []
        
        results = []
        for file in daily_files:
            filepath = os.path.join(self.daily_dir, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取条目（按 ## 分割）
                    entries = content.split('\n## ')
                    for entry in entries[1:]:  # 跳过第一个空条目
                        if memory_type is None or f"**Type**: {memory_type}" in entry:
                            results.append(entry.strip())
                            if len(results) >= n:
                                return results
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        return results
    
    async def get_unindexed(self, indexed_hashes: Dict[str, str]) -> List[MemoryEntry]:
        """获取未索引的条目：遍历所有Markdown文件，解析条目，计算hash并与indexed_hashes比较"""
        unindexed = []
        
        # 遍历daily目录
        try:
            daily_files = os.listdir(self.daily_dir)
        except FileNotFoundError:
            daily_files = []
        
        for file in daily_files:
            if not file.endswith('.md'):
                continue
            filepath = os.path.join(self.daily_dir, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析条目（按 ## 分割）
                sections = content.split('\n## ')
                for section in sections:
                    if not section.strip():
                        continue
                    lines = section.split('\n')
                    if not lines:
                        continue
                    
                    # 第一行是时间戳
                    timestamp_line = lines[0].strip()
                    # 提取元数据
                    source = "unknown"
                    mem_type = "unknown"
                    content_lines = []
                    for line in lines[1:]:
                        if line.startswith('**Source**:'):
                            source = line.replace('**Source**:', '').strip()
                        elif line.startswith('**Type**:'):
                            mem_type = line.replace('**Type**:', '').strip()
                        else:
                            content_lines.append(line)
                    
                    entry_content = '\n'.join(content_lines).strip()
                    if not entry_content:
                        continue
                    
                    # 计算hash
                    entry_hash = hashlib.md5(entry_content.encode()).hexdigest()
                    
                    # 检查是否已索引
                    found = False
                    for mem_id, h in indexed_hashes.items():
                        if h == entry_hash:
                            found = True
                            break
                    
                    if not found:
                        # 创建MemoryEntry
                        entry = MemoryEntry(
                            content=entry_content,
                            timestamp=timestamp_line,
                            source=source,
                            memory_type=mem_type,
                            metadata={}
                        )
                        unindexed.append(entry)
            
            except Exception as e:
                logger.error(f"Error parsing {filepath}: {e}")
        
        return unindexed