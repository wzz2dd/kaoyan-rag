"""
检索优化模块

负责实现高效的混合检索策略，结合向量语义检索和BM25关键词检索，
使用RRF算法进行结果融合重排，提高检索的准确性和召回率
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================================
# 异常类定义
# ============================================================================

class RetrievalError(Exception):
    """检索异常"""
    pass


class FilterError(Exception):
    """过滤异常"""
    pass


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class RetrievalConfig:
    """检索配置"""

    # 向量检索配置
    vector_k: int = 10                   # 向量检索返回数量
    search_type: str = "similarity"      # 搜索类型

    # BM25检索配置
    bm25_k: int = 10                     # BM25检索返回数量

    # RRF配置
    rrf_k: int = 60                      # RRF平滑参数

    # 最终结果配置
    default_top_k: int = 10              # 默认返回数量

    # 性能配置
    parallel_search: bool = True         # 是否并行检索


# ============================================================================
# 主类
# ============================================================================

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        config: Optional[RetrievalConfig | Dict[str, Any]] = None
    ):
        """
        初始化检索优化模块

        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
            config: 可选的配置对象或字典
        """
        if config is not None:
            if isinstance(config, RetrievalConfig):
                self.config = config
            else:
                self.config = RetrievalConfig(**config)
        else:
            self.config = RetrievalConfig()

        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None
        self._retrieval_stats: Dict[str, Any] = {}

        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.vector_k}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.config.bm25_k
        )

        logger.info(f"检索器设置完成 (vector_k={self.config.vector_k}, bm25_k={self.config.bm25_k})")

    def vector_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        向量检索

        Args:
            query: 查询文本
            k: 返回结果数量，默认使用配置值

        Returns:
            检索到的文档列表
        """
        if k is None:
            k = self.config.vector_k

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.debug(f"向量检索返回 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def bm25_search(self, query: str) -> List[Document]:
        """
        BM25检索

        Args:
            query: 查询文本

        Returns:
            检索到的文档列表
        """
        try:
            docs = self.bm25_retriever.invoke(query)
            logger.debug(f"BM25检索返回 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []

    def hybrid_search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认使用配置值

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = self.config.default_top_k

        # 并行或串行执行检索
        if self.config.parallel_search:
            vector_docs, bm25_docs = self._parallel_search(query)
        else:
            vector_docs = self.vector_retriever.invoke(query)
            bm25_docs = self.bm25_retriever.invoke(query)

        # 处理检索失败的情况
        if not vector_docs and not bm25_docs:
            logger.warning(f"查询 '{query}' 无任何检索结果")
            return []

        if not vector_docs:
            logger.warning("向量检索无结果，仅使用BM25结果")
            return bm25_docs[:top_k]

        if not bm25_docs:
            logger.warning("BM25检索无结果，仅使用向量检索结果")
            return vector_docs[:top_k]

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)

        # 更新统计信息
        self._retrieval_stats['last_query'] = query
        self._retrieval_stats['vector_count'] = len(vector_docs)
        self._retrieval_stats['bm25_count'] = len(bm25_docs)
        self._retrieval_stats['merged_count'] = len(reranked_docs)

        return reranked_docs[:top_k]

    def _parallel_search(self, query: str) -> tuple:
        """
        并行执行向量检索和BM25检索

        Args:
            query: 查询文本

        Returns:
            (向量检索结果, BM25检索结果)
        """
        vector_docs = []
        bm25_docs = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self.vector_retriever.invoke, query)
            future_bm25 = executor.submit(self.bm25_retriever.invoke, query)

            for future in as_completed([future_vector, future_bm25]):
                try:
                    if future == future_vector:
                        vector_docs = future.result()
                    else:
                        bm25_docs = future.result()
                except Exception as e:
                    logger.error(f"并行检索异常: {e}")

        return vector_docs, bm25_docs

    def metadata_filtered_search(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[Document]:
        """
        带元数据过滤的检索

        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量

        Returns:
            过滤后的文档列表
        """
        if not filters:
            logger.warning("过滤条件为空，执行普通混合检索")
            return self.hybrid_search(query, top_k)

        # 先进行混合检索，获取更多候选
        candidate_count = max(top_k * 3, 15)
        docs = self.hybrid_search(query, candidate_count)

        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            if self._match_filters(doc, filters):
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        logger.info(f"元数据过滤: {len(docs)} -> {len(filtered_docs)} 个文档")

        return filtered_docs

    def _match_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """
        检查文档是否匹配过滤条件

        Args:
            doc: 文档对象
            filters: 过滤条件

        Returns:
            是否匹配
        """
        for key, value in filters.items():
            if key not in doc.metadata:
                return False

            doc_value = doc.metadata[key]

            # 支持列表值（OR逻辑）
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            else:
                if doc_value != value:
                    return False

        return True

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: Optional[int] = None
    ) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        if k is None:
            k = self.config.rrf_k

        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score #doc_scores[doc_id]若已有值，则返回值，若没有，则返回0

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'config': {
                'vector_k': self.config.vector_k,
                'bm25_k': self.config.bm25_k,
                'rrf_k': self.config.rrf_k,
                'default_top_k': self.config.default_top_k,
                'parallel_search': self.config.parallel_search,
            },
            'last_retrieval': self._retrieval_stats.copy(),
        }
        return stats

    def get_retrievers(self) -> Dict[str, Any]:
        """
        获取检索器对象

        Returns:
            包含向量检索器和BM25检索器的字典
        """
        return {
            'vector_retriever': self.vector_retriever,
            'bm25_retriever': self.bm25_retriever,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def retrieve(
    query: str,
    vectorstore: FAISS,
    chunks: List[Document],
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    检索主入口

    Args:
        query: 查询文本
        vectorstore: FAISS向量存储
        chunks: 文档块列表
        filters: 可选的元数据过滤条件
        top_k: 返回结果数量
        config: 可选配置项

    Returns:
        检索到的文档列表
    """
    retrieval_module = RetrievalOptimizationModule(
        vectorstore=vectorstore,
        chunks=chunks,
        config=config
    )

    if filters:
        return retrieval_module.metadata_filtered_search(query, filters, top_k)
    else:
        return retrieval_module.hybrid_search(query, top_k)
