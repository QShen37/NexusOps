from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config

class DocumentSplitterService:
    """文档分割服务 - 使用Langchain进行分割"""
    def __init__(self):
        # 初始化文档分割服务
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap
        self.chunk_min_size = config.chunk_min_size

        # Markdown标题分割器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")
                # 不按三级标题分割，以防过度碎片化
            ],
            strip_headers=False, # 将标题保留在内容中
        )

        # 递归字符分割器（用于二次分割，使用更大的Chunk Size）
        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2, # 加倍chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成，chunk_size = {self.chunk_size}, "
            f"secondary_chunk_size = {self.chunk_overlap}, "
            f"overlap = {self.chunk_overlap}, "
        )

    def _merge_small_chunk(self, document: List[Document], min_size: int) -> List[Document]:
        if not document:
            return []

        merge_docs = []
        current_doc = None

        for doc in document:
            doc_size = len(doc.page_content)
            if current_doc is None:
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < min_size:
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                merge_docs.append(current_doc)
                current_doc = doc
        return merge_docs

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown文件内容为空：{file_path}")
            return []

        try:
            # 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 按大小进一步分割
            docs_after_split = self.textSplitter.split_documents(md_docs)

            # 合并太小的分片（<300字符）
            final_docs = self._merge_small_chunk(docs_after_split, self.chunk_min_size)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown分割完成：{file_path} -> {len(final_docs)}个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown文件内容为空：{file_path}")
            return []
        try:
            # 直接采用递归字符分割器
            docs = self.textSplitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.debug(f"文本分割完成：{file_path} -> {len(docs)}个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败：{file_path}，错误：{e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
                智能分割文档 (根据文件类型选择分割器)

                Args:
                    content: 文档内容
                    file_path: 文件路径

                Returns:
                    List[Document]: 文档分片列表
        """
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

document_splitter_service = DocumentSplitterService()