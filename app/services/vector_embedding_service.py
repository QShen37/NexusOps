"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口"""
# 对Embedding中的文件查询嵌入以及文本查询嵌入进行重写
from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.config import config


class LocalEmbeddings(Embeddings):
    """本地 SentenceTransformer Embeddings

    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            device: str = "cpu",  # 可选: "cuda", "cpu"
    ):
        """
        初始化本地 Embeddings

        Args:
            model_name: SentenceTransformer 模型名称
            device: 运行设备 (cpu/cuda)
        """
        try:
            logger.info(f"正在加载 SentenceTransformer 模型: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            self.device = device

            # 获取向量维度
            self.dimensions = config.EMBEDDING_DIMENSION

            logger.info(
                f"本地 Embeddings 初始化完成 - "
                f"模型: {model_name}, 维度: {self.dimensions}, 设备: {device}"
            )
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"模型加载失败: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档列表 (LangChain 标准接口)

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档")

            # 批量编码文本
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,  # 转换为 numpy 数组
                show_progress_bar=False  # 不显示进度条（避免日志混乱）
            )

            # 转换为 list 格式
            embeddings_list = embeddings.tolist()

            logger.debug(f"批量嵌入完成, 维度: {len(embeddings_list[0])}")
            return embeddings_list

        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本 (LangChain 标准接口)

        Args:
            text: 查询文本

        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")

        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")

            # 编码查询文本
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # 转换为 list 格式
            embedding_list = embedding.tolist()

            logger.debug(f"查询嵌入完成, 维度: {len(embedding_list)}")
            return embedding_list

        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e


# 全局单例
vector_embedding_service = LocalEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    device=config.device
)