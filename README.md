# 项目名称
NexusOps – 面向智能运维的多智能体 Agent 框架

## 项目简介
本项目设计并实现了一套基于大语言模型（LLM）的智能运维系统 NexusOps，结合 RAG、Tool Calling、MCP、CLI与多智能体协作机制，实现从知识检索、任务规划到故障诊断与命令执行的完整 AI 运维闭环。
## 代码结构
清晰展示项目目录结构，标注核心文件/文件夹作用
```
NexusOps/
├── app/                    # 主应用目录
│   ├── agent/             # Agent 核心模块（入口）
│   ├── api/               # API 路由/接口层
│   ├── core/              # 核心业务逻辑
│   ├── models/            # 数据模型定义
│   ├── services/          # 业务服务层
│   ├── tools/             # 工具函数/辅助工具
│   ├── config.py          # 配置文件
│   └── main.py            # 应用主入口
├── mcp_server/            # MCP 服务器模块
│   ├── cls_server.py      # CLS（日志服务）服务器
│   ├── monitor_server.py  # 监控服务器
│   └── system_server.py   # 系统服务器
├── static/                # 静态资源目录
│   ├── app.js             # 前端 JavaScript 逻辑
│   ├── cli.html           # CLI 辅助页面
│   ├── index.html         # 主页面
│   └── style.css          # 样式表
├── .gitignore             # Git 忽略文件配置（包含 .idea/）
├── README.md              # 项目说明文档
└── requirements.txt       # Python 依赖清单
```

## 核心模块
根据项目结构，各核心模块功能描述如下：

### 1. app/agent —— Agent 核心入口模块
负责 Agent 的初始化、生命周期管理和任务调度，实现 Agent 的启动、运行和关闭逻辑，是整个系统的控制中枢。

### 2. app/api —— API 接口层
处理外部 HTTP 请求，提供 RESTful API 接口，负责请求参数校验、响应格式化，将外部调用转发给 `core` 层处理。

### 3. app/core —— 核心业务逻辑层
实现项目的核心算法和业务规则，处理主要业务流程编排，是系统最关键的逻辑处理中心。

### 4. app/models —— 数据模型层
定义数据库表结构、数据实体和 DTO（数据传输对象），负责数据的结构化表示和类型定义。

### 5. app/services —— 业务服务层
协调多个 `core` 模块完成复杂业务操作，封装业务用例，处理事务逻辑和外部服务调用。

### 6. app/tools —— 工具函数库
提供通用工具函数（如时间处理、字符串格式化、加密解密等），封装可复用的辅助方法，供其他模块调用。

### 7. app/config.py —— 配置管理模块
统一管理应用配置（支持环境变量、配置文件等多种来源），提供全局配置对象的访问接口。

### 8. app/main.py —— 应用启动入口
负责应用启动前的初始化工作（加载配置、连接数据库、注册路由等），启动 Web 服务器或 Agent 主循环。

### 9. mcp_server/ —— MCP 服务集群
- **cls_server.py**：提供日志服务（CLS）的查询、分析和处理能力
- **monitor_server.py**：实现系统监控功能，收集性能指标和告警数据
- **system_server.py**：管理系统资源和基础设施，提供服务发现和健康检查

### 10. static/ —— 前端静态资源
提供 Web 界面的 UI 组件，包括页面结构（HTML）、样式定义（CSS）和交互逻辑（JavaScript），实现终端用户的可视化操作界面。


## 环境准备
1. 安装Python（推荐3.13+）
2. 安装项目依赖
```bash
pip install -r requirements.txt
```

## 执行方式
### 1. 基础运行
```bash
# 启动3个server
python .\mcp_server\cls_server.py
python .\mcp_server\monitor_server.py
python .\mcp_server\system_server.py

# 在9900端口启动交互界面
python -m uvicorn app.main:app --host 127.0.0.1 --port 9900
```

## 注意事项
- 运行前需在`config/`目录配置相关参数（主要是API）
- 数据文件需放置在指定目录或者根据页面交互上传

## 待办事项
- 将cli确认转移到前端
- skills实现
- 实现一个一键启动的bat文件

## 联系方式
作者：XXX
邮箱：XXX