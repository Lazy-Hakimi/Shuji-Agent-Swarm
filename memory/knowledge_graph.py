"""
枢机 (Shuji) - 知识图谱
基于NetworkX实现，支持实体关系抽取和图遍历
"""
import json
import os
import hashlib
from typing import List, Dict, Optional, Any, Tuple, Set
import logging

logger = logging.getLogger(__name__)

# 尝试导入spaCy进行NER（可选）
try:
    import spacy
    NLP_AVAILABLE = True
    # 加载小型英文模型
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("spaCy not installed. Using simple keyword-based entity extraction.")


class KnowledgeGraph:
    """知识图谱基类"""
    
    async def add_node(
        self,
        node_id: str,
        content: str,
        node_type: str = "entity",
        metadata: Optional[Dict] = None
    ):
        """添加节点"""
        raise NotImplementedError
    
    async def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0
    ):
        """添加边"""
        raise NotImplementedError
    
    async def get_related(self, content: str, k: int = 5) -> List[str]:
        """获取相关内容（通过图遍历）"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        raise NotImplementedError


class NetworkXKnowledgeGraph(KnowledgeGraph):
    """NetworkX知识图谱"""
    
    def __init__(self, graph_path: str = "./knowledge_graph.json"):
        self.graph_path = graph_path
        self.available = False
        self.graph = None
        
        try:
            import networkx as nx
            self.nx = nx
            
            # 加载或创建图
            if os.path.exists(graph_path):
                with open(graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
            else:
                self.graph = nx.DiGraph()
            
            self.available = True
            
        except ImportError:
            logger.error("networkx not installed. Knowledge graph disabled.")
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
    
    async def add_node(
        self,
        node_id: str,
        content: str,
        node_type: str = "entity",
        metadata: Optional[Dict] = None
    ):
        """添加节点"""
        if not self.available:
            return
        
        self.graph.add_node(
            node_id,
            content=content,
            type=node_type,
            metadata=metadata or {}
        )
        
        # 提取实体并创建关系
        await self._extract_and_link(node_id, content)
        
        # 保存
        self._save()
    
    async def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0
    ):
        """添加边"""
        if not self.available:
            return
        
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            weight=weight
        )
        
        self._save()
    
    async def get_related(self, content: str, k: int = 5) -> List[str]:
        """
        获取相关内容：通过实体匹配找到相关节点，然后BFS遍历邻居
        """
        if not self.available or self.graph.number_of_nodes() == 0:
            return []
        
        # 找到与内容相关的节点（通过实体匹配）
        matched_nodes = self._find_matched_nodes(content)
        if not matched_nodes:
            return []
        
        # BFS遍历获取邻居节点内容
        related_contents = set()
        for node_id in matched_nodes:
            # 获取节点本身内容
            node_data = self.graph.nodes[node_id]
            related_contents.add(node_data.get('content', ''))
            
            # BFS深度为2
            visited = {node_id}
            queue = [(node_id, 0)]
            while queue:
                current, depth = queue.pop(0)
                if depth >= 2:  # 限制深度
                    continue
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if depth + 1 <= 2:
                            neighbor_data = self.graph.nodes[neighbor]
                            related_contents.add(neighbor_data.get('content', ''))
                            queue.append((neighbor, depth + 1))
        
        # 返回前k个
        return list(related_contents)[:k]
    
    def _find_matched_nodes(self, content: str) -> List[str]:
        """根据内容中的实体找到匹配的节点"""
        matched = []
        content_lower = content.lower()
        
        # 提取内容中的实体（使用NER或关键词）
        entities = self._extract_entities(content)
        
        for node_id, data in self.graph.nodes(data=True):
            node_content = data.get('content', '').lower()
            # 检查实体匹配
            for entity in entities:
                if entity.lower() in node_content:
                    matched.append(node_id)
                    break
            # 如果没找到实体，尝试单词匹配
            if node_id not in matched:
                for word in content_lower.split():
                    if len(word) > 3 and word in node_content:
                        matched.append(node_id)
                        break
        
        return list(set(matched))  # 去重
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的实体"""
        if NLP_AVAILABLE:
            doc = nlp(text)
            return [ent.text for ent in doc.ents]
        else:
            # 简单关键词：大写单词或专有名词启发式
            words = text.split()
            entities = []
            for word in words:
                # 假设大写开头的可能是实体
                if word and word[0].isupper() and len(word) > 1:
                    entities.append(word)
                # 或者连续两个大写字母的缩写
                if word.isupper() and len(word) <= 5:
                    entities.append(word)
            return list(set(entities))
    
    async def _extract_and_link(self, node_id: str, content: str):
        """提取节点内容中的实体并与现有节点建立关系"""
        entities = self._extract_entities(content)
        
        for entity in entities:
            # 查找包含该实体的已有节点
            for existing_id, data in self.graph.nodes(data=True):
                if existing_id == node_id:
                    continue
                if entity.lower() in data.get('content', '').lower():
                    # 创建双向关系
                    self.graph.add_edge(
                        node_id,
                        existing_id,
                        relation="mentions",
                        weight=0.5
                    )
                    self.graph.add_edge(
                        existing_id,
                        node_id,
                        relation="mentioned_by",
                        weight=0.5
                    )
    
    def _save(self):
        """保存图谱"""
        if not self.available:
            return
        
        try:
            data = self.nx.node_link_data(self.graph)
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.available:
            return {"available": False}
        
        return {
            "available": True,
            "type": "networkx",
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
        }


class SimpleKnowledgeGraph(KnowledgeGraph):
    """简单知识图谱（用于测试）"""
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, str]] = []
    
    async def add_node(
        self,
        node_id: str,
        content: str,
        node_type: str = "entity",
        metadata: Optional[Dict] = None
    ):
        """添加节点"""
        self.nodes[node_id] = {
            "content": content,
            "type": node_type,
            "metadata": metadata or {}
        }
    
    async def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0
    ):
        """添加边"""
        self.edges.append((source, target, relation))
    
    async def get_related(self, content: str, k: int = 5) -> List[str]:
        """获取相关内容"""
        results = []
        for node_id, data in self.nodes.items():
            if any(word in data['content'].lower() for word in content.lower().split()):
                results.append(data['content'])
                if len(results) >= k:
                    break
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "available": True,
            "type": "simple",
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }