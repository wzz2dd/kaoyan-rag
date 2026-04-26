"""
索引构建模块

负责将招生信息文档块转化为向量表示，并构建高效的FAISS向量索引
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================================
# 异常类定义
# ============================================================================

class IndexBuildError(Exception):
    """索引构建异常"""
    pass


class ModelLoadError(Exception):
    """模型加载异常"""
    pass


class IndexSaveError(Exception):
    """索引保存异常"""
    pass


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class IndexConfig:
    """索引构建配置"""

    # 嵌入模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    normalize_embeddings: bool = True

    # 索引存储配置
    index_save_path: str = "./vector_index"

    # 检索配置
    default_top_k: int = 5


# ============================================================================
# 主类
# ============================================================================

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
        device: str = "cpu",
        config: Optional[IndexConfig | Dict[str, Any]] = None
    ):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
            device: 运行设备 ('cpu' 或 'cuda')
            config: 可选的配置对象或字典
        """
        if config is not None:
            if isinstance(config, IndexConfig):
                self.config = config
            else:
                self.config = IndexConfig(**config)
        else:
            self.config = IndexConfig(
                embedding_model=model_name,
                index_save_path=index_save_path,
                device=device
            )

        self.model_name = self.config.embedding_model
        self.index_save_path = self.config.index_save_path
        self.device = self.config.device
        self.embeddings = None
        self.vectorstore = None
        self._index_stats: Dict[str, Any] = {}

        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        logger.info(f"运行设备: {self.device}")

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': self.config.normalize_embeddings}
            )
            logger.info("嵌入模型初始化完成")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise ModelLoadError(f"无法加载嵌入模型 {self.model_name}: {e}")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引

        Args:
            chunks: 文档块列表

        Returns:
            FAISS向量存储对象

        Raises:
            IndexBuildError: 构建索引失败时抛出
        """
        logger.info("正在构建FAISS向量索引...")

        if not chunks:
            raise IndexBuildError("文档块列表不能为空")

        try:
            # 构建FAISS向量存储
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )

            # 记录统计信息
            self._index_stats = {
                'total_vectors': len(chunks),
                'embedding_model': self.model_name,
                'device': self.device
            }

            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            return self.vectorstore

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise IndexBuildError(f"构建向量索引失败: {e}")

    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档

        Args:
            new_chunks: 新的文档块列表

        Raises:
            IndexBuildError: 索引不存在时抛出
        """
        if not self.vectorstore:
            raise IndexBuildError("请先构建向量索引")

        if not new_chunks:
            logger.warning("新文档块列表为空，跳过添加")
            return

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")

        try:
            self.vectorstore.add_documents(new_chunks)
            self._index_stats['total_vectors'] = self._index_stats.get('total_vectors', 0) + len(new_chunks)
            logger.info("新文档添加完成")
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise IndexBuildError(f"添加文档失败: {e}")

    def save_index(self):
        """
        保存向量索引到配置的路径

        Raises:
            IndexSaveError: 保存失败时抛出
        """
        if not self.vectorstore:
            raise IndexSaveError("请先构建向量索引")

        try:
            # 确保保存目录存在
            Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

            self.vectorstore.save_local(self.index_save_path)
            logger.info(f"向量索引已保存到: {self.index_save_path}")

        except PermissionError as e:
            logger.error(f"保存索引权限不足: {e}")
            raise IndexSaveError(f"保存索引权限不足: {e}")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise IndexSaveError(f"保存索引失败: {e}")

    def load_index(self) -> Optional[FAISS]:
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表

        Raises:
            IndexBuildError: 索引不存在时抛出
        """
        if not self.vectorstore:
            raise IndexBuildError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        带分数的相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            (文档, 分数) 元组列表，分数越小表示越相似

        Raises:
            IndexBuildError: 索引不存在时抛出
        """
        if not self.vectorstore:
            raise IndexBuildError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'model_name': self.model_name,
            'device': self.device,
            'index_save_path': self.index_save_path,
            'index_exists': self.vectorstore is not None,
        }

        if self.vectorstore:
            # FAISS 索引信息
            try:
                index = self.vectorstore.index
                stats['total_vectors'] = index.ntotal
                stats['vector_dimension'] = index.d
            except Exception:
                stats['total_vectors'] = self._index_stats.get('total_vectors', 'unknown')
                stats['vector_dimension'] = 'unknown'

        return stats

    def get_vectorstore(self) -> Optional[FAISS]:
        """
        获取向量存储对象

        Returns:
            FAISS向量存储对象，如果不存在返回None
        """
        return self.vectorstore

    def delete_index(self):
        """
        删除内存中的索引
        """
        self.vectorstore = None
        self._index_stats = {}
        logger.info("内存中的索引已清除")


# ============================================================================
# 便捷函数
# ============================================================================

def build_and_save_index(
    chunks: List[Document],
    config: Optional[Dict[str, Any]] = None
) -> FAISS:
    """
    索引构建主入口

    Args:
        chunks: 文档块列表
        config: 可选配置项

    Returns:
        构建完成的FAISS向量存储
    """
    index_builder = IndexConstructionModule(config=config)
    vectorstore = index_builder.build_vector_index(chunks)
    index_builder.save_index()
    return vectorstore
