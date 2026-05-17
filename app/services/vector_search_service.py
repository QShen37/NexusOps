"""向量检索模块
使用 PyMilvus 原生连接，支持自动重连
"""

from typing import Any, Dict, List

from loguru import logger
from pymilvus import Collection, connections

from app.services.QueryRewriter import QueryRewriter
from app.services.vector_embedding_service import vector_embedding_service


class SearchResult:
    """搜索结果"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """向量检索服务"""

    def __init__(self):
        self._connected = False
        self._collection = None
        self._collection_name = None
        logger.info("向量检索服务初始化完成")

    def _ensure_connection(self) -> bool:
        """确保 Milvus 连接已建立"""
        try:
            # 检查是否已连接
            if connections.has_connection("default"):
                self._connected = True
                return True

            # 尝试连接
            from app.config import config

            logger.info(f"连接 Milvus: {config.milvus_host}:{config.milvus_port}")
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=config.milvus_port,
            )

            # 获取 collection
            if connections.has_connection("default"):
                self._collection_name = config.collection_name
                self._collection = Collection(self._collection_name)
                self._connected = True
                logger.info(
                    f"Milvus 连接成功，collection: {self._collection_name}, "
                    f"实体数量: {self._collection.num_entities}"
                )
                return True

        except Exception as e:
            logger.warning(f"Milvus 连接失败: {e}")
            self._connected = False
            return False

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        检索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 确保连接
            if not self._ensure_connection():
                logger.warning("Milvus 未连接，返回空结果")
                return []

            logger.info(f"开始检索相似文档，查询: {query}，topK: {top_k}")

            # 查询文本向量化
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功，维度：{len(query_vector)}")

            # 检查 collection 是否为空
            if self._collection.is_empty:
                logger.warning("Collection 为空，没有可检索的数据")
                return []

            # 构建搜索参数
            search_params = {
                "metric_type": "L2",  # 欧氏距离
                "params": {"nprobe": 10},
            }

            # 执行搜索（注意：参数名是 param 不是 params）
            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,  # 关键修改：param 而不是 params
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            # 解析检索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"搜索完成，找到 {len(search_results)} 个相似文档")
            return search_results

        except Exception as e:
            logger.error(f"搜索相似文档失败：{e}")
            return []  # 返回空列表而不是抛出异常

    def search_similar_documents_rewrite(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        检索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 确保连接
            if not self._ensure_connection():
                logger.warning("Milvus 未连接，返回空结果")
                return []

            logger.info(f"开始检索相似文档，查询: {query}，topK: {top_k}")

            # 查询文本向量化
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功，维度：{len(query_vector)}")

            # 检查 collection 是否为空
            if self._collection.is_empty:
                logger.warning("Collection 为空，没有可检索的数据")
                return []

            # 构建搜索参数
            search_params = {
                "metric_type": "L2",  # 欧氏距离
                "params": {"nprobe": 10},
            }

            # 执行搜索（注意：参数名是 param 不是 params）
            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,  # 关键修改：param 而不是 params
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            # 解析检索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"搜索完成，找到 {len(search_results)} 个相似文档")
            return search_results

        except Exception as e:
            logger.error(f"搜索相似文档失败：{e}")
            return []  # 返回空列表而不是抛出异常


# 全局单例
vector_search_service = VectorSearchService()