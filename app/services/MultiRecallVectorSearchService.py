"""向量检索模块 - 支持多路召回
使用 PyMilvus 原生连接，支持自动重连和多路召回策略
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re
import asyncio

from loguru import logger
from pymilvus import Collection, connections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.vector_embedding_service import vector_embedding_service


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    recall_source: str = "vector"  # 召回来源: vector, keyword, hybrid, rerank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "recall_source": self.recall_source,
        }


@dataclass
class RecallConfig:
    """召回配置"""
    vector_weight: float = 0.8  # 提高向量权重 (从0.6提高到0.8)
    keyword_weight: float = 0.2  # 降低关键词权重 (从0.4降低到0.2)
    enable_keyword: bool = True
    enable_hybrid: bool = True
    enable_rerank: bool = False  # 默认关闭重排序，避免引入额外错误
    top_k_vector: int = 10
    top_k_keyword: int = 5  # 减少关键词召回数量
    top_k_final: int = 3


class KeywordRetriever:
    """关键词检索器 - 基于 TF-IDF 和 BM25"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None
        )
        self.documents: List[str] = []
        self.tfidf_matrix = None
        self._fitted = False

    def fit(self, documents: List[str]):
        """拟合 TF-IDF 模型"""
        if not documents:
            return
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self._fitted = True
        logger.info(f"关键词检索器拟合完成，文档数: {len(documents)}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """关键词检索"""
        if not self._fitted or not self.documents:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

            # 添加相似度阈值过滤，低于阈值的不要
            threshold = 0.1
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > threshold:
                    results.append((int(idx), float(score)))
            return results
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def add_documents(self, documents: List[str]):
        self.documents.extend(documents)
        self._fitted = False
        self.fit(self.documents)


class HybridRetriever:
    """混合检索器 - 结合向量检索和关键词检索"""

    def __init__(self, vector_weight: float = 0.8, keyword_weight: float = 0.2):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.keyword_retriever = KeywordRetriever()

    def fit_keyword_index(self, documents: List[str]):
        self.keyword_retriever.fit(documents)

    def search(
            self,
            query: str,
            query_vector: List[float],
            vector_results: List[Tuple[int, SearchResult, float]],
            top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        改进的混合检索：使用 RRF 风格融合，提高向量检索权重
        """
        # 1. 关键词检索（减少数量）
        keyword_results = self.keyword_retriever.search(query, top_k=top_k)

        # 2. 分数融合 - 使用排名融合而非分数加权
        fusion_scores = defaultdict(float)
        recall_sources = {}

        # 向量检索 - 给予更高权重，使用排名倒数
        if vector_results:
            for rank, (idx, result, score) in enumerate(vector_results[:top_k], 1):
                # 使用排名倒数作为分数，排名越靠前分数越高
                rank_score = 1.0 / rank
                fusion_scores[idx] += rank_score * self.vector_weight
                recall_sources[idx] = "vector"

        # 关键词检索 - 降低权重
        if keyword_results:
            for rank, (idx, score) in enumerate(keyword_results[:top_k], 1):
                rank_score = 1.0 / rank
                fusion_scores[idx] += rank_score * self.keyword_weight
                if idx not in recall_sources:
                    recall_sources[idx] = "keyword"
                else:
                    recall_sources[idx] = "hybrid"

        # 3. 排序返回
        sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)

        return [(idx, score, recall_sources.get(idx, "unknown"))
                for idx, score in sorted_results[:top_k]]


class Reranker:
    """重排序器 - 使用 LLM 或交叉编码器对结果重排序"""

    def __init__(self, llm=None):
        self.llm = llm

    async def rerank(
            self,
            query: str,
            results: List[SearchResult],
            top_k: int = 3
    ) -> List[SearchResult]:
        if len(results) <= top_k:
            return results
        if self.llm:
            return await self._llm_rerank(query, results, top_k)
        return self._cross_encoder_rerank(query, results, top_k)

    async def _llm_rerank(
            self,
            query: str,
            results: List[SearchResult],
            top_k: int
    ) -> List[SearchResult]:
        """使用 LLM 重排序 - 简化版，避免 LLM 调用失败"""
        # 暂时返回原始排序，避免 LLM 调用出错
        return results[:top_k]

    def _cross_encoder_rerank(
            self,
            query: str,
            results: List[SearchResult],
            top_k: int
    ) -> List[SearchResult]:
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [(query, r.content[:512]) for r in results]
            scores = model.predict(pairs)
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [r for r, _ in scored_results[:top_k]]
        except Exception as e:
            logger.warning(f"交叉编码器重排序失败: {e}")
            return results[:top_k]


class MultiRecallVectorSearchService:
    """多路召回向量检索服务"""

    def __init__(self):
        self._connected = False
        self._collection = None
        self._collection_name = None
        self._documents_cache: List[str] = []
        self._documents_metadata: List[Dict] = []

        # 初始化多路召回组件 - 提高向量权重
        self.hybrid_retriever = HybridRetriever(vector_weight=0.8, keyword_weight=0.2)
        self.reranker = Reranker()
        self.recall_config = RecallConfig()

        logger.info("多路召回向量检索服务初始化完成")

    def _ensure_connection(self) -> bool:
        """确保 Milvus 连接已建立"""
        try:
            if connections.has_connection("default"):
                self._connected = True
                if self._collection is None:
                    from app.config import config
                    self._collection_name = config.collection_name
                    self._collection = Collection(self._collection_name)
                return True

            from app.config import config
            logger.info(f"连接 Milvus: {config.milvus_host}:{config.milvus_port}")
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=config.milvus_port,
            )

            if connections.has_connection("default"):
                self._collection_name = config.collection_name
                self._collection = Collection(self._collection_name)
                self._connected = True
                logger.info(
                    f"Milvus 连接成功，collection: {self._collection_name}, "
                    f"实体数量: {self._collection.num_entities if self._collection else 0}"
                )
                self._load_document_cache()
                return True
            else:
                logger.warning("Milvus 连接失败：无法建立连接")
                return False

        except Exception as e:
            logger.warning(f"Milvus 连接失败: {e}")
            self._connected = False
            self._collection = None
            return False

    def _load_document_cache(self):
        """加载文档缓存（用于关键词检索）"""
        try:
            if not self._collection:
                logger.warning("Collection 未初始化，无法加载文档缓存")
                return

            if self._collection.is_empty:
                logger.warning("Collection 为空，无需加载文档缓存")
                return

            results = self._collection.query(
                expr="id != ''",
                output_fields=["content", "metadata"],
                limit=5000
            )

            self._documents_cache = [r.get("content", "") for r in results]
            self._documents_metadata = [r.get("metadata", {}) for r in results]

            if self._documents_cache:
                self.hybrid_retriever.fit_keyword_index(self._documents_cache)
                logger.info(f"文档缓存加载完成: {len(self._documents_cache)} 条")
            else:
                logger.warning("文档缓存为空")
        except Exception as e:
            logger.warning(f"加载文档缓存失败: {e}")
            self._documents_cache = []
            self._documents_metadata = []

    def _vector_search(
            self,
            query_vector: List[float],
            top_k: int = 10
    ) -> List[Tuple[int, SearchResult, float]]:
        """改进的纯向量检索 - 使用更好的分数转换"""
        try:
            if self._collection is None:
                logger.warning("Collection 未初始化，无法执行向量检索")
                return []

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            vector_results = []
            for hits in results:
                for hit in hits:
                    # 改进：L2距离转相似度 (距离越小，相似度越高)
                    # 公式: similarity = 1 / (1 + distance)  范围 (0, 1]
                    similarity = 1.0 / (1.0 + hit.distance)

                    # 进一步提高分数，让向量检索结果更突出
                    # 给向量检索一个基础加成
                    boosted_similarity = similarity * 1.2

                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=boosted_similarity,
                        metadata=hit.entity.get("metadata", {}),
                        recall_source="vector"
                    )
                    idx = self._find_document_index(result.content)
                    if idx >= 0:
                        vector_results.append((idx, result, boosted_similarity))

            return vector_results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _find_document_index(self, content: str) -> int:
        """查找文档在缓存中的索引"""
        for i, doc in enumerate(self._documents_cache):
            if doc == content:
                return i
        return -1

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """关键词检索 - 添加置信度过滤"""
        results = self.hybrid_retriever.keyword_retriever.search(query, top_k)
        # 过滤低分结果
        return [(idx, score) for idx, score in results if score > 0.15]

    def _hybrid_search(
            self,
            query: str,
            query_vector: List[float],
            top_k: int = 10
    ) -> List[Tuple[int, SearchResult, float, str]]:
        """
        改进的混合检索
        """
        # 1. 向量检索（获取更多候选）
        vector_results = self._vector_search(query_vector, top_k=top_k * 2)

        # 2. 混合检索
        hybrid_results = self.hybrid_retriever.search(
            query=query,
            query_vector=query_vector,
            vector_results=vector_results,
            top_k=top_k
        )

        # 3. 构建结果
        final_results = []
        for idx, score, source in hybrid_results:
            if 0 <= idx < len(self._documents_cache):
                result = SearchResult(
                    id=str(idx),
                    content=self._documents_cache[idx],
                    score=score,
                    metadata=self._documents_metadata[idx] if idx < len(self._documents_metadata) else {},
                    recall_source=source
                )
                final_results.append((idx, result, score, source))

        return final_results

    async def search_similar_documents(
            self,
            query: str,
            top_k: int = 3,
            recall_type: str = "vector",  # 默认改为 vector，这是最稳定的
            config: Optional[RecallConfig] = None
    ) -> List[SearchResult]:
        """
        多路召回检索相似文档

        建议使用 recall_type="vector" 获得最佳效果
        """
        if config:
            self.recall_config = config

        try:
            if not self._ensure_connection():
                logger.warning("Milvus 未连接，返回空结果")
                return []

            logger.info(f"开始多路召回检索，查询: {query[:50]}...，类型: {recall_type}，topK: {top_k}")

            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功，维度：{len(query_vector)}")

            if self._collection is None:
                logger.warning("Collection 未初始化，返回空结果")
                return []

            if self._collection.is_empty:
                logger.warning("Collection 为空，没有可检索的数据")
                return []

            if not self._documents_cache:
                logger.warning("文档缓存为空，尝试重新加载")
                self._load_document_cache()
                if not self._documents_cache:
                    logger.warning("文档缓存仍为空，返回空结果")
                    return []

            results = []

            if recall_type == "vector":
                # 纯向量检索 - 最稳定，推荐使用
                vector_results = self._vector_search(query_vector, top_k=top_k)
                results = [r for _, r, _ in vector_results[:top_k]]

            elif recall_type == "keyword":
                keyword_results = self._keyword_search(query, top_k=top_k)
                for idx, score in keyword_results:
                    if 0 <= idx < len(self._documents_cache):
                        result = SearchResult(
                            id=str(idx),
                            content=self._documents_cache[idx],
                            score=score,
                            metadata=self._documents_metadata[idx] if idx < len(self._documents_metadata) else {},
                            recall_source="keyword"
                        )
                        results.append(result)

            elif recall_type == "hybrid":
                hybrid_results = self._hybrid_search(query, query_vector, top_k=top_k)
                results = [r for _, r, _, _ in hybrid_results[:top_k]]

            elif recall_type == "rerank":
                hybrid_results = self._hybrid_search(query, query_vector, top_k=top_k * 2)
                candidates = [r for _, r, _, _ in hybrid_results]
                if candidates:
                    results = await self.reranker.rerank(query, candidates, top_k)

            elif recall_type == "multi":
                results = await self._multi_channel_recall(query, query_vector, top_k)

            else:
                # 默认使用向量检索
                vector_results = self._vector_search(query_vector, top_k=top_k)
                results = [r for _, r, _ in vector_results[:top_k]]

            logger.info(f"搜索完成，找到 {len(results)} 个相似文档")
            return results

        except Exception as e:
            logger.error(f"搜索相似文档失败：{e}")
            import traceback
            traceback.print_exc()
            return []

    async def _multi_channel_recall(
            self,
            query: str,
            query_vector: List[float],
            top_k: int
    ) -> List[SearchResult]:
        """多路并行召回"""
        # 简单起见，直接使用向量检索
        vector_results = self._vector_search(query_vector, top_k=top_k)
        return [r for _, r, _ in vector_results[:top_k]]

    def update_document_cache(self):
        """更新文档缓存（当文档有变化时调用）"""
        self._load_document_cache()


# 全局单例
multi_recall_vector_search_service = MultiRecallVectorSearchService()