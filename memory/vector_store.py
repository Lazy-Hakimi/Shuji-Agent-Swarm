"""
枢机 (Shuji) - 向量存储
基于ChromaDB实现，使用asyncio.to_thread避免阻塞
"""
import asyncio
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储基类"""
    
    async def add(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Optional[List[float]]:
        """添加向量"""
        raise NotImplementedError
    
    async def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """搜索向量"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB向量存储（异步封装）"""
    
    def __init__(
        self,
        collection_name: str = "shuji_memory",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.available = False
        self.collection = None
        self.client = None
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            self.chroma = chromadb
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            # 初始化客户端（同步操作，但在__init__中执行一次）
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_func
            )
            self.available = True
        except ImportError:
            logger.warning("chromadb not installed. Vector search disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
    
    async def add(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Optional[List[float]]:
        """添加向量（在线程池中执行）"""
        if not self.available:
            return None
        
        def _add():
            self.collection.add(
                ids=[id],
                documents=[text],
                metadatas=[metadata or {}]
            )
            # ChromaDB不返回embedding，我们可以通过查询获得，但这里不需要
            return None
        
        try:
            return await asyncio.to_thread(_add)
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            return None
    
    async def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """搜索向量（在线程池中执行）"""
        if not self.available:
            return []
        
        def _search():
            return self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_dict
            )
        
        try:
            results = await asyncio.to_thread(_search)
            
            # 格式化结果
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.available:
            return {"available": False}
        
        try:
            count = self.collection.count()
            return {
                "available": True,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "document_count": count,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"available": False, "error": str(e)}


class SimpleVectorStore(VectorStore):
    """简单内存向量存储（用于测试）"""
    
    def __init__(self):
        self.vectors: Dict[str, Any] = {}  # 存储embedding向量
        self.texts: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.available = True
    
    async def add(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Optional[List[float]]:
        """添加向量（使用简单的hash embedding）"""
        # 简单的embedding：字符频率向量
        embedding = self._simple_embed(text)
        self.vectors[id] = embedding
        self.texts[id] = text
        self.metadata[id] = metadata or {}
        return embedding.tolist()
    
    async def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """搜索向量"""
        if not self.vectors:
            return []
        
        query_embedding = self._simple_embed(query)
        
        # 计算相似度
        similarities = {}
        for id, vec in self.vectors.items():
            # 检查过滤条件
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    if self.metadata[id].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # 计算余弦相似度
            import numpy as np
            similarity = np.dot(query_embedding, vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-8
            )
            similarities[id] = similarity
        
        # 排序并返回top-k
        sorted_ids = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for id, similarity in sorted_ids:
            results.append({
                "id": id,
                "content": self.texts[id],
                "metadata": self.metadata[id],
                "distance": 1 - similarity,
            })
        
        return results
    
    def _simple_embed(self, text: str) -> Any:
        """简单embedding (字符频率)"""
        import numpy as np
        vec = np.zeros(256)
        for char in text.lower():
            vec[ord(char) % 256] += 1
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "available": True,
            "type": "simple",
            "document_count": len(self.vectors),
        }