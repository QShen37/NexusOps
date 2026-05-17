# NexusOps（更新中...）

> 面向智能运维的多智能体 Agent 框架

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

NexusOps 是一套基于大语言模型（LLM）的智能运维系统，结合 **RAG**、**Tool Calling**、**MCP**、**CLI** 与**多智能体协作机制**，实现从知识检索、任务规划到故障诊断与命令执行的完整 AI 运维闭环。

### ✨ 核心特性

- 🔍 **知识检索增强**：基于 RAG 的智能知识库
- 🛠️ **工具调用能力**：灵活的 Tool Calling 机制
- 🤖 **多智能体协作**：协同完成复杂运维任务
- 📊 **实时监控**：全方位的系统监控能力
- 🖥️ **可视化界面**：友好的 Web 交互界面

---

## 📁 代码结构

```
NexusOps/
├── 📂 app/ # 主应用目录
│ ├── 🧠 agent/ # Agent 核心模块（入口）
│ ├── 🌐 api/ # API 路由/接口层
│ ├── ⚙️ core/ # 核心业务逻辑
│ ├── 📊 models/ # 数据模型定义
│ ├── 🔧 services/ # 业务服务层
│ ├── 🛠️ tools/ # 工具函数/辅助工具
│ ├── ⚡ config.py # 配置文件
│ └── 🚀 main.py # 应用主入口
│
├── 🔌 mcp_server/ # MCP 服务器模块
│ ├── 📋 cls_server.py # CLS（日志服务）服务器
│ ├── 📈 monitor_server.py # 监控服务器
│ └── 💻 system_server.py # 系统服务器
│
├── 🎨 static/ # 静态资源目录
│ ├── app.js # 前端 JavaScript 逻辑
│ ├── cli.html # CLI 辅助页面
│ ├── index.html # 主页面
│ └── style.css # 样式表
│
├── 🔒 .gitignore # Git 忽略文件配置
├── 📝 README.md # 项目说明文档
└── 📦 requirements.txt # Python 依赖清单
```


---

## 🧩 核心模块

### 🎯 应用核心层

| 模块 | 职责 | 说明 |
|:-----|:-----|:-----|
| `app/agent` | Agent 核心入口 | 负责 Agent 初始化、生命周期管理和任务调度，系统控制中枢 |
| `app/main.py` | 应用入口 | 负责应用初始化，启动 Web 服务器或 Agent 主循环 |
| `app/config.py` | 配置管理 | 统一管理应用配置，支持环境变量和配置文件 |

### 🔌 接口与业务层

| 模块 | 职责 | 说明 |
|:-----|:-----|:-----|
| `app/api` | API 接口层 | 处理 HTTP 请求，提供 RESTful API，负责参数校验与响应格式化 |
| `app/core` | 核心业务逻辑 | 实现核心算法和业务规则，处理主要业务流程编排 |
| `app/services` | 业务服务层 | 协调多个 core 模块，封装复杂业务操作和事务逻辑 |

### 📦 数据与工具层

| 模块 | 职责 | 说明 |
|:-----|:-----|:-----|
| `app/models` | 数据模型层 | 定义数据结构、实体和 DTO，负责数据的结构化表示 |
| `app/tools` | 工具函数库 | 提供通用工具函数，封装可复用的辅助方法 |

### 🌐 服务与资源层

| 模块 | 职责 | 说明 |
|:-----|:-----|:-----|
| `mcp_server/` | MCP 服务集群 | 提供日志、监控、系统管理等基础服务 |
| `static/` | 前端静态资源 | 提供 Web 界面 UI 组件，实现可视化操作 |

---

### 📡 MCP 服务详情

| 服务文件 | 功能描述 |
|:---------|:---------|
| `cls_server.py` | 日志服务查询、分析和处理 |
| `monitor_server.py` | 系统监控、性能指标和告警 |
| `system_server.py` | 资源管理、服务发现和健康检查 |

---

## 🛠️ 环境准备

### 系统要求
- Python 3.13+
- conda 环境
- pip 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/QShen37/NexusOps.git
cd NexusOps
```

2. **虚拟环境配置（需要保证电脑有Minicoda）**

```bash
conda create -n nexus_ops python=3.13
conda activate nexus_ops
```

3. **安装依赖**
```bash
pip install -r requirement.txt
```

4. **执行方式（一键启动/关闭）**
```bash
# 一键启动服务

.\manager\start.bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 9900

# 打开浏览器访问：http://127.0.0.1:9900

# 一键关闭服务（需自行关闭端口）
.\manager\stop.bat
```

## 注意事项
- 运行前需在`config_template.py`配置相关参数并且修改名字为`config.py`（主要是API）
- 数据文件需放置在指定目录或者根据页面交互上传

## 🎯 RAG 系统评测

本项目集成了完整的 RAG 评测系统（基于 Ragas 框架），支持 6 个核心指标的自动化评测。

### 快速评测

```bash
# 1. 安装评测依赖
pip install -r app/services/ragas_requirements.txt

# 2. 运行快速评测（使用 10 个预设问题）
python run_quick_eval.py
```

### 评测指标

- **Faithfulness** (忠实度): 答案是否基于检索到的上下文
```bash
# 步骤1: 从答案中提取核心主张
claims = [
    "CPU使用率过高需要先获取当前时间",
    "然后查询系统日志",
    "分析CPU消耗进程", 
    "常见原因是死循环或流量突增"
]

# 步骤2: 逐条验证是否能从上下文推断
supported = 0
for claim in claims:
    prompt = f"上下文: {context}\n主张: {claim}\n这个主张能从上下文中推断出来吗？回答是/否"
    if llm_response == "是":
        supported += 1

# 步骤3: 计算比例
faithfulness = supported / len(claims) = 4/4 = 1.0
```
- **Answer Relevancy** (相关性): 答案是否直接回答问题
```bash
# 步骤1: 使用语义模型计算相似度
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

question_embedding = model.encode("CPU使用率过高告警如何排查？")
answer_embedding = model.encode("CPU使用率过高需要先获取当前时间，然后查询系统日志...")

# 步骤2: 计算余弦相似度
similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
# 结果: 0.72
```
- **Context Recall** (召回率): 检索到的上下文完整性
```bash
# 步骤1: 从标准答案中提取核心信息
ground_truth_claims = [
    "获取当前时间",
    "查询系统日志", 
    "分析CPU消耗进程",
    "死循环或流量突增原因"
]

# 步骤2: 检查每个信息是否在上下文中
found = 0
for claim in ground_truth_claims:
    if claim in context_text:  # 或 LLM 判断
        found += 1

context_recall = found / len(ground_truth_claims) = 4/4 = 1.0
```
- **Answer Similarity** (相似度): 答案与标准答案的语义相似度
```bash
# 使用 Sentence-BERT 语义模型
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

answer_emb = model.encode("CPU使用率过高需要先获取当前时间，然后查询系统日志...")
gt_emb = model.encode("CPU使用率过高告警处理方案：1. 获取当前时间...")

similarity = cosine_similarity([answer_emb], [gt_emb])[0][0]
# 结果: 0.82
```


```bash
  📊 评估对比汇总
══════════════════════════════════════════════════════════════════════
  指标                             基础 RAG      增强 RAG        提升
──────────────────────────────────────────────────────────────────────
  faithfulness                     0.9111       0.9221       ↑ 0.0110
  answer_relevancy                 0.7514       0.7549       ↑ 0.0035
  context_recall                   0.8444       0.9157       ↑ 0.0713
  answer_similarity                0.7949       0.8076       ↑ 0.0127
══════════════════════════════════════════════════════════════════════

```

---

## 待办事项
- ~~rag添加rewrite和hybrid混合索引~~
- rag添加图片以及pdf上传（预防pdf是图片扫描版本）功能
- ~~rag添加ragas测评（已完成）~~
- ~~将cli确认转移到前端（已完成）~~
- ~~skills实现（已完成）~~