"""
配置管理
"""

from typing import Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # 应用配置
    app_name: str = "AI Operation and Maintenance Agent"
    app_version: str = "1.0.0"
    app_description: str = "AI Operation and Maintenance Agent"
    device: str = "cpu"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 9900

    # API配置
    DEEPSEEK_API_KEY: str = "sk-582412a410e44bf0b431d30b5aa21f63"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEk_MODEL: str = "deepseek-chat"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = SentenceTransformer(EMBEDDING_MODEL, device=device).get_embedding_dimension()

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000 # ms
    collection_name: str = "biz"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    upload_data_path: Path = BASE_DIR / "upload_data"

    # RAG配置
    rag_top_k: int = 3
    rag_model: str = "deepseek-chat"

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100
    chunk_min_size: int = 300

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://127.0.0.1:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://127.0.0.1:8004/mcp"
    mcp_system_transport: str = "streamable-http"
    mcp_system_url: str = "http://127.0.0.1:8005/mcp"

    @property
    def mcp_server(self) -> Dict[str, Dict[str, Any]]:
        # 获取完整的MCP服务器配置
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
            "system": {
                "transport": self.mcp_system_transport,
                "url": self.mcp_system_url,
            }
        }

# 全局配置实例
config = Settings()