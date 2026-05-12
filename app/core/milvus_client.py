"""Milvus 客户端模块"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
    MilvusException
)

from app.config import config

def _patch_pymilvus_milvus_client_orm_alias() -> None:
    """
    langchain_milvus 内部创建的 MilvusClient 会将 _using 设为 ``cm-{id}``，
    该别名未在 pymilvus.orm.connections 中注册；随后 ORM ``Collection(..., using=...)``
    会抛出 ConnectionNotExistException: should create connection first.

    在已通过 ``connections.connect(alias="default", ...)`` 建立连接后，
    强制让 MilvusClient 使用 ``default`` 别名，与 ORM 一致。
    """
    if getattr(_patch_pymilvus_milvus_client_orm_alias, "_done", False):
        return
    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient
    except ImportError:
        return

    _orig_init = MilvusClient.__init__

    def _wrapped_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        self._using = "default"

    MilvusClient.__init__ = _wrapped_init  # type: ignore[method-assign]
    setattr(_patch_pymilvus_milvus_client_orm_alias, "_done", True)


class MilvusClientManager:
    """Milvus 客户端管理器

    负责管理 Milvus 数据库的连接、collection 的创建和维护。
    提供连接管理、健康检查、资源清理等功能，确保与 Milvus 的交互安全可靠。

    主要功能：
    - 自动创建和初始化 collection
    - 管理连接生命周期（连接、断开、重连）
    - 向量维度一致性检查和自动修复
    - 支持上下文管理器（with 语句）
    """

    # ==================== 常量定义 ====================
    # 这些常量定义了 collection 的核心配置，修改时需谨慎

    """Milvus 中 collection 的名称，用于存储业务知识数据"""
    COLLECTION_NAME: str = config.collection_name
    VECTOR_DIM: int = config.EMBEDDING_DIMENSION
    """主键 ID 字段的最大长度（字符数）"""
    ID_MAX_LENGTH: int = 100
    """content 字段的最大长度（字符数），用于存储原始文本内容"""
    CONTENT_MAX_LENGTH: int = 8000
    """collection 的分片数量，用于分布式存储和查询"""
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器

        初始化时将客户端和 collection 设置为 None，
        实际的连接建立需要在调用 connect() 方法时进行。
        """
        self._client: MilvusClient | None = None
        """MilvusClient 实例，用于执行向量操作（插入、搜索等）"""

        self._collection: Collection | None = None
        """Collection 实例，用于管理 collection 的元数据和索引"""

    # ==================== 公开方法 ====================

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        此方法会执行以下步骤：
        1. 检查是否已连接（幂等性保证）
        2. 建立与 Milvus 服务器的连接
        3. 检查 collection 是否存在
        4. 如果不存在则创建 collection 和索引
        5. 如果存在则验证向量维度是否匹配
        6. 加载 collection 到内存中

        Returns:
            MilvusClient: Milvus 客户端实例，可用于执行向量操作

        Raises:
            RuntimeError: 连接或初始化失败时抛出，包含具体错误信息

        Note:
            - 此方法是幂等的，多次调用不会导致重复连接
            - 如果检测到向量维度不匹配，会自动删除并重建 collection
            - 连接超时时间由配置中的 milvus_timeout 决定（毫秒）
        """
        # 幂等性检查：如果已经连接且 collection 已初始化，直接返回现有客户端
        # 这样可以避免导入阶段因多个模块初始化导致重复连接
        if self._collection is not None and self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            # 修复 pymilvus 库中可能存在的 ORM 别名问题
            # 这是为了避免某些版本中出现的兼容性问题
            _patch_pymilvus_milvus_client_orm_alias()

            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 建立与 Milvus 服务器的连接
            # alias="default": 为连接设置别名，方便后续引用
            # timeout: 连接超时时间（秒），从配置中获取
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,  # 转换为秒
            )

            # 创建 Milvus 客户端，用于后续操作
            # URI 格式: http://host:port
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # ========== Collection 初始化 ==========
            # 检查并创建 collection
            if not self._collection_exists():
                # Collection 不存在，创建新的
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                # Collection 已存在，检查维度一致性
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)

                # 获取现有 collection 的 schema（数据结构定义）
                schema = self._collection.schema
                vector_field = None
                existing_dim = None

                # 遍历所有字段，找到向量字段并获取其维度
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break

                # 检查向量维度是否与当前配置一致
                # pymilvus 中向量维度存储在字段的 params 字典中
                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        # 维度不匹配！需要删除旧的 collection 并重新创建
                        # 这种情况通常发生在更换 Embedding 模型后
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        logger.info(f"正在删除旧 collection '{self.COLLECTION_NAME}'...")
                        _ = utility.drop_collection(self.COLLECTION_NAME)
                        logger.info(f"正在重新创建 collection '{self.COLLECTION_NAME}'...")
                        self._create_collection()
                        logger.info(f"成功重新创建 collection，维度: {self.VECTOR_DIM}")
                    else:
                        # 维度匹配，无需额外操作
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 加载 collection 到内存中
            # 只有加载后的 collection 才能进行查询操作
            self._load_collection()

            return self._client

        except MilvusException as e:
            # Milvus 特定的异常，如查询失败、连接错误等
            logger.error(f"Milvus 操作失败: {e}")
            self.close()  # 清理资源
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            # 网络连接相关的异常
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            # 其他未预期的异常
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        通过此方法可以获取 Collection 对象，直接操作 Milvus 的高级功能，
        如创建索引、执行复杂查询等。

        Returns:
            Collection: collection 实例，可用于高级操作

        Raises:
            RuntimeError: collection 未初始化时抛出，通常是因为未调用 connect() 方法

        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """
        健康检查

        向 Milvus 服务器发送轻量级请求，验证服务是否正常响应。
        可用于监控系统定期检查服务状态。

        Returns:
            bool: True 表示服务健康，False 表示服务异常

        Note:
            此方法不会抛出异常，所有错误都会被捕获并返回 False，
            适合在监控脚本中使用。
        """
        try:
            if self._client is None:
                return False

            # 尝试列出所有连接，这是一个轻量级操作
            # 如果成功返回，说明 Milvus 服务正常响应
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """
        关闭连接并释放资源

        执行以下清理操作：
        1. 释放 collection 占用的内存
        2. 断开与 Milvus 服务器的连接
        3. 清空内部引用

        注意：
        - 关闭后如果需要再次使用，必须重新调用 connect()
        - 即使某些步骤失败，也会继续执行后续清理
        - 所有错误会被收集并记录，不会抛出异常
        """
        errors = []  # 收集所有清理过程中的错误

        # 步骤1：释放 collection 的内存占用
        # 这会将 collection 从内存中卸载，但不会删除数据
        try:
            if self._collection is not None:
                self._collection.release()  # 释放内存
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        # 步骤2：断开与 Milvus 服务器的连接
        try:
            if connections.has_connection("default"):
                connections.disconnect("default")  # 断开连接
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        # 清空客户端引用
        self._client = None

        # 如果有错误发生，记录日志
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    # ==================== 上下文管理器支持 ====================

    def __enter__(self) -> "MilvusClientManager":
        """
        上下文管理器入口

        支持使用 with 语句自动管理连接生命周期。
        进入 with 代码块时会自动调用 connect() 建立连接。

        Returns:
            MilvusClientManager: 返回自身实例，供 with 语句使用

        """
        _ = self.connect()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: object
    ) -> None:
        """
        上下文管理器退出

        无论 with 代码块中是否发生异常，都会自动调用 close() 释放资源。

        Args:
            exc_type: 异常类型，如果没有异常则为 None
            exc_val: 异常实例，如果没有异常则为 None
            exc_tb: 异常追踪信息，如果没有异常则为 None

        Note:
            - 即使有异常发生，也会确保资源被正确释放
            - 不会抑制异常，异常会正常向上传播
        """
        self.close()

    # ==================== 私有方法 ====================

    def _collection_exists(self) -> bool:
        """
        检查 collection 是否存在

        使用 Milvus 的 utility.has_collection 方法检查 collection 是否已创建。

        Returns:
            bool: True 表示 collection 存在，False 表示不存在

        Note:
            pymilvus 的类型标注可能不准确，实际返回 bool 类型，
            这里显式转换为 bool 确保类型安全。
        """
        # utility.has_collection 返回类型可能为 bool 或 Boolean，统一转换为 bool
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)  # type: ignore[arg-type]

    def _create_collection(self) -> None:
        """
        创建 biz collection

        执行以下步骤：
        1. 定义 collection 的字段结构（schema）
        2. 创建 collection 实例
        3. 为向量字段创建索引以加速搜索

        Collection 字段说明：
        - id: VARCHAR，主键，用于唯一标识每条记录
        - vector: FLOAT_VECTOR，向量嵌入，用于相似度搜索
        - content: VARCHAR，原始文本内容
        - metadata: JSON，灵活的元数据存储，可存储任意 JSON 格式的附加信息

        Note:
            enable_dynamic_field=False 禁用了动态字段，
            这要求插入的数据只能包含预定义的字段，有助于保证数据一致性。
        """
        # ========== 定义字段结构 ==========
        fields = [
            # 主键字段：记录的唯一标识符
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,  # 可变长度字符串类型
                max_length=self.ID_MAX_LENGTH,  # 最大长度限制
                is_primary=True,  # 设为主键，保证唯一性
            ),
            # 向量字段：存储文本的向量嵌入
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,  # 浮点数向量类型
                dim=self.VECTOR_DIM,  # 向量维度，必须与 Embedding 模型匹配
            ),
            # 内容字段：存储原始文本
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,  # 限制文本长度，防止过大的数据
            ),
            # 元数据字段：存储附加信息
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,  # JSON 类型，可灵活存储结构化数据
                # 例如：{"source": "web", "author": "张三", "timestamp": "2024-01-01"}
            ),
        ]

        # ========== 创建 Schema ==========
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",  # collection 描述信息
            enable_dynamic_field=False,  # 禁止动态添加字段，保持数据结构固定
        )

        # ========== 创建 Collection ==========
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,  # 分片数，用于水平扩展
        )

        # ========== 创建索引 ==========
        # 索引对于向量搜索至关重要，可以大幅提升查询性能
        self._create_index()

    def _create_index(self) -> None:
        """
        为 vector 字段创建索引

        索引参数说明：
        - metric_type: "L2" 欧氏距离，常用的相似度度量方式
          其他可选：IP（内积）、COSINE（余弦相似度）
        - index_type: "IVF_FLAT" 索引类型
          IVF_FLAT 是 IVF（倒排文件索引）的变体，适合中等规模的数据集
        - nlist: 聚类中心数量，影响搜索精度和速度的平衡

        关于索引类型选择：
        - IVF_FLAT: 精度高，适合数据量在百万级以下
        - IVF_SQ8: 压缩索引，节省内存，精度略有损失
        - HNSW: 速度极快，适合实时查询，但索引构建慢
        - GPU_IVF_FLAT: 使用 GPU 加速，需要 Milvus GPU 版本

        Note:
            此方法要求在 _collection 已初始化的情况下调用，
            且必须在插入数据之前创建索引，否则每次插入都会触发索引重建。
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        # 定义索引参数
        index_params = {
            "metric_type": "L2",  # 使用欧氏距离计算相似度
            "index_type": "IVF_FLAT",  # 索引类型：倒排文件索引
            "params": {"nlist": 128},  # 聚类数量，128 是适合中小数据集的默认值
        }

        # 为 vector 字段创建索引
        # create_index 返回的结果在没有错误时通常为 None
        _ = self._collection.create_index(
            field_name="vector",  # 为向量字段创建索引
            index_params=index_params,
        )

        logger.info("成功为 vector 字段创建索引")

    def _load_collection(self) -> None:
        """
        加载 collection 到内存

        使用 load() 方法将 collection 加载到内存中。
        只有加载后的 collection 才能进行查询和搜索操作。

        加载内存说明：
        - 加载后，Milvus 会将数据索引加载到内存
        - 查询时直接从内存读取，速度更快
        - 大数据集需要较大的内存空间

        兼容性处理：
        由于 pymilvus 不同版本加载状态检查方法不同，
        这里尝试两种方式：
        1. 新版本：使用 utility.load_state() 检查状态
        2. 旧版本：直接 load()，捕获 "already loaded" 异常

        Raises:
            MilvusException: 加载失败时抛出
            Exception: 其他未预期的异常
        """
        if self._collection is None:
            # 如果 collection 实例不存在，从 Milvus 获取
            self._collection = Collection(self.COLLECTION_NAME)

        # 尝试检查 collection 是否已加载（兼容不同版本）
        try:
            # 方式1：新版本 pymilvus 使用 utility.load_state
            load_state = utility.load_state(self.COLLECTION_NAME)
            # load_state 可能返回字符串或枚举，统一转换为字符串比较
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                # 未加载，执行加载操作
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                # 已加载，无需重复加载
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方式2：旧版本不支持 load_state，尝试直接加载
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                # 检查是否是"已加载"的异常
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    # collection 已经加载，这是正常情况
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    # 其他异常，向上抛出
                    raise
        except Exception as e:
            # 其他未预期的异常
            logger.error(f"加载 collection 失败: {e}")
            raise


# ==================== 全局单例 ====================
# 创建全局唯一的 Milvus 客户端管理器实例
# 这种单例模式确保整个应用共享同一个连接，避免资源浪费
milvus_manager = MilvusClientManager()