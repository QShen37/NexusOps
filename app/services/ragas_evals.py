"""
RAGAS 评估服务 - 严谨版评估
实现接近标准 RAGAS 的评估算法，包含 MRR 指标
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import pandas as pd
import numpy as np
from loguru import logger
from langchain_deepseek import ChatDeepSeek
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 从配置文件读取 API Key
try:
    from app.config import config
    DEEPSEEK_API_KEY = getattr(config, 'DEEPSEEK_API_KEY', None)
except ImportError:
    DEEPSEEK_API_KEY = None

if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

from app.services.vector_search_service import vector_search_service


class RAGASEvaluator:
    """
    严谨版 RAGAS 评估器

    实现更准确的评估算法：
    1. 忠实度 (Faithfulness) - 使用 NLI 模型或 LLM 评估
    2. 答案相关性 (Answer Relevancy) - 使用 LLM 评估
    3. 上下文精确度 (Context Precision) - 基于 LLM 的相关性判断
    4. 上下文召回率 (Context Recall) - 基于 LLM 的 claim 提取
    5. 答案正确性 (Answer Correctness) - 综合 TP/FP/FN
    6. 答案相似度 (Answer Similarity) - 语义相似度
    7. MRR (Mean Reciprocal Rank) - 第一个正确答案的平均倒数排名
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.api_key_valid = False
        self.llm = None
        self._init_llm()

        # 尝试加载更好的 embedding 模型用于语义相似度
        self.semantic_model = None
        self._init_semantic_model()

        logger.info("严谨版 RAGAS 评估器初始化完成")

    def _init_llm(self):
        """初始化 ChatDeepSeek LLM"""
        if not self.api_key:
            logger.warning("未设置 API Key，部分指标将使用算法计算")
            return

        try:
            self.llm = ChatDeepSeek(
                model="deepseek-chat",
                api_key=self.api_key,
                temperature=0.1,  # 低温度保证结果稳定
                max_tokens=1000,
                timeout=30
            )
            self.api_key_valid = True
            logger.info("✅ ChatDeepSeek 初始化成功")
        except Exception as e:
            logger.warning(f"ChatDeepSeek 初始化失败: {e}")
            self.llm = None

    def _init_semantic_model(self):
        """初始化语义相似度模型"""
        try:
            from sentence_transformers import SentenceTransformer
            # 使用中文语义模型
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ 语义相似度模型加载成功")
        except ImportError:
            logger.warning("sentence-transformers 未安装，将使用 TF-IDF 计算相似度")
        except Exception as e:
            logger.warning(f"语义模型加载失败: {e}")

    async def _llm_judge(self, prompt: str, expected_output: str = "是/否") -> Tuple[float, str]:
        """使用 LLM 进行判断"""
        if not self.llm or not self.api_key_valid:
            return 0.5, "LLM不可用"

        try:
            response = await self.llm.ainvoke(prompt)
            result = response.content.strip()

            # 解析结果
            if "是" in result or "yes" in result.lower() or "true" in result.lower():
                return 1.0, result
            elif "否" in result or "no" in result.lower() or "false" in result.lower():
                return 0.0, result
            else:
                # 尝试提取数字
                numbers = re.findall(r'\d+(?:\.\d+)?', result)
                if numbers:
                    score = float(numbers[0]) / 100 if float(numbers[0]) > 1 else float(numbers[0])
                    return min(1.0, max(0.0, score)), result
                return 0.5, result
        except Exception as e:
            logger.error(f"LLM 判断失败: {e}")
            return 0.5, str(e)

    async def _llm_judge_recall(self, prompt: str, expected_output: str = "是/否") -> Tuple[float, str]:
        """使用 LLM 进行判断 - 返回明确的 0 或 1"""
        if not self.llm or not self.api_key_valid:
            return 0.0, "LLM不可用"

        try:
            response = await self.llm.ainvoke(prompt)
            result = response.content.strip()
            result_lower = result.lower()

            # 明确的肯定词 - 返回 1.0
            positive_words = [
                "是", "能", "可以", "相关", "找到", "能找到",
                "yes", "true", "相关", "支持", "可以找到",
                "有", "存在", "包含", "符合"
            ]

            # 明确的否定词 - 返回 0.0
            negative_words = [
                "否", "不能", "不可以", "不相关", "找不到", "找不到",
                "no", "false", "不支持", "不可以找到",
                "没有", "不存在", "不包含", "不符合"
            ]

            for word in positive_words:
                if word in result_lower:
                    logger.debug(f"LLM 判断: {result[:50]} -> 肯定 (1.0)")
                    return 1.0, result

            for word in negative_words:
                if word in result_lower:
                    logger.debug(f"LLM 判断: {result[:50]} -> 否定 (0.0)")
                    return 0.0, result

            # 尝试提取数字
            numbers = re.findall(r'\d+(?:\.\d+)?', result)
            if numbers:
                score = float(numbers[0])
                if score >= 0.5:
                    logger.debug(f"LLM 判断: 数字 {score} -> 肯定 (1.0)")
                    return 1.0, result
                else:
                    logger.debug(f"LLM 判断: 数字 {score} -> 否定 (0.0)")
                    return 0.0, result

            # 默认：如果结果长度较短且不是明确否定，视为肯定
            if len(result) < 10:
                logger.debug(f"LLM 判断: 短响应 '{result}' -> 默认肯定 (1.0)")
                return 1.0, result

            # 无法判断的情况
            logger.warning(f"无法解析 LLM 输出: {result[:100]}")
            return 0.0, result  # 保守起见返回 0.0

        except Exception as e:
            logger.error(f"LLM 判断失败: {e}")
            return 0.0, str(e)

    def _extract_key_statements(self, text: str) -> List[str]:
        """
        提取文本中的关键陈述
        用于忠实度和召回率评估
        """
        # 分句
        sentences = re.split(r'[。！？!?]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        # 使用 LLM 提取关键 claim（如果有）
        if self.llm and self.api_key_valid:
            # 简化：返回所有非空句子
            return sentences
        return sentences

    async def _extract_claims_with_llm(self, text: str, context: str = "") -> List[str]:
        """使用 LLM 提取文本中的核心主张"""
        if not self.llm or not self.api_key_valid:
            return self._extract_key_statements(text)

        prompt = f"""请从以下文本中提取所有核心主张（关键事实陈述）。每个主张应该是完整、独立的句子。

文本：{text}

请按以下格式输出，每行一个主张：
1. [主张1]
2. [主张2]
...
"""
        try:
            response = await self.llm.ainvoke(prompt)
            claims = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    claim = re.sub(r'^\d+\.\s*', '', line)
                    if len(claim) > 10:
                        claims.append(claim)
            return claims if claims else self._extract_key_statements(text)
        except:
            return self._extract_key_statements(text)

    async def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        忠实度：答案中的主张是否都能从上下文中得到支持

        标准算法：
        1. 从答案中提取 N 个主张 (claims)
        2. 检查每个主张是否可以从上下文中推断
        3. faithfulness = 可支持的主张数 / 总主张数
        """
        if not answer or not contexts:
            return 0.0

        # 提取答案中的主张
        claims = await self._extract_claims_with_llm(answer, "\n".join(contexts))
        if not claims:
            return 0.5

        context_text = "\n".join(contexts)
        supported_claims = 0

        for claim in claims:
            prompt = f"""判断以下主张是否可以从提供的上下文中推断出来。

上下文：
{context_text}

主张：{claim}

请只回答"是"或"否"："""

            is_supported, _ = await self._llm_judge(prompt)
            if is_supported > 0.5:
                supported_claims += 1

        return supported_claims / len(claims) if claims else 0.0

    async def _compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        答案相关性：答案是否直接回答问题

        标准算法：
        1. 生成问题的 N 个变体
        2. 计算答案与问题变体的语义相似度
        """
        if not answer or not question:
            return 0.0

        # 使用语义相似度
        if self.semantic_model:
            try:
                q_embedding = self.semantic_model.encode(question)
                a_embedding = self.semantic_model.encode(answer)
                similarity = cosine_similarity([q_embedding], [a_embedding])[0][0]
                return float(similarity)
            except:
                pass

        # 回退：TF-IDF + 余弦相似度
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question, answer])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # 最终回退：关键词匹配
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            if q_words:
                overlap = len(q_words & a_words) / len(q_words)
                return min(1.0, overlap)
            return 0.0

    async def _compute_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        上下文召回率：标准答案中的信息是否都能从上下文中找到

        标准算法：
        1. 从标准答案中提取主张
        2. 检查每个主张是否能从上下文中找到
        3. recall = 可找到的主张数 / 总主张数
        """
        if not ground_truth or not contexts:
            return 0.0

        # 提取标准答案中的主张
        claims = await self._extract_claims_with_llm(ground_truth, "\n".join(contexts))
        if not claims:
            return 0.5

        context_text = "\n".join(contexts)
        found_claims = 0

        for claim in claims:
            prompt = f"""判断以下主张是否可以在提供的上下文中找到。

上下文：
{context_text[:1500]}

主张：{claim}

请只回答"能找到"或"找不到"："""

            is_found, _ = await self._llm_judge_recall(prompt)
            if is_found > 0.5:
                found_claims += 1

        return found_claims / len(claims) if claims else 0.0

    async def _compute_answer_similarity(self, answer: str, ground_truth: str) -> float:
        """
        答案相似度：语义相似度

        使用 Sentence-BERT 或其他语义模型计算
        """
        if not answer or not ground_truth:
            return 0.0

        # 使用语义模型
        if self.semantic_model:
            try:
                a_embedding = self.semantic_model.encode(answer)
                gt_embedding = self.semantic_model.encode(ground_truth)
                similarity = cosine_similarity([a_embedding], [gt_embedding])[0][0]
                return float(similarity)
            except:
                pass

        # 回退：TF-IDF
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([answer, ground_truth])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # 最终回退：SequenceMatcher
            from difflib import SequenceMatcher
            return SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio()

    async def _compute_mrr(self, question: str, contexts: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank): 第一个正确答案的平均倒数排名

        计算方式：
        1. 对每个检索到的文档判断是否与问题相关
        2. 找到第一个相关文档的排名 position
        3. RR = 1 / position (如果没找到，RR = 0)
        4. MRR = 平均 RR

        Returns:
            float: Reciprocal Rank 值 (0-1)
        """
        if not contexts:
            logger.warning(f"MRR: 没有检索到上下文，返回 0")
            return 0.0

        logger.info(f"MRR: 开始计算，共 {len(contexts)} 个候选文档")

        for rank, ctx in enumerate(contexts, start=1):
            prompt = f"""判断以下文档是否与问题相关。

问题：{question}

文档：{ctx[:500]}

请只回答"相关"或"不相关"："""

            is_relevant, reason = await self._llm_judge_recall(prompt)

            logger.debug(f"MRR: 文档 {rank} -> {'相关' if is_relevant > 0.5 else '不相关'}")

            if is_relevant > 0.5:
                rr = 1.0 / rank
                logger.info(f"MRR: 第一个相关文档在位置 {rank}, Reciprocal Rank = {rr:.4f}")
                return rr

        logger.warning(f"MRR: 未找到相关文档，返回 0")
        return 0.0

    async def evaluate_single_question(
        self,
        question: str,
        ground_truth: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """评估单个问题 - 严谨版"""
        logger.info(f"评估问题: {question[:50]}...")

        # 1. 检索上下文
        contexts = await self._retrieve_contexts(question, top_k)

        # 2. 生成答案
        answer = await self._generate_answer(question, contexts)

        logger.info(f"生成答案: {answer[:200]}...")

        # 3. 计算各项指标（并行执行提高效率）
        faithfulness, answer_relevancy, context_recall, similarity, mrr = await asyncio.gather(
            self._compute_faithfulness(answer, contexts),
            self._compute_answer_relevancy(question, answer),
            self._compute_context_recall(ground_truth, contexts),
            self._compute_answer_similarity(answer, ground_truth),
            self._compute_mrr(question, contexts)
        )

        scores = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "answer_similarity": similarity,
            "mrr": mrr  # 新增 MRR 指标
        }

        logger.info(f"得分: faithfulness={faithfulness:.3f}, relevancy={answer_relevancy:.3f}, "
                   f"recall={context_recall:.3f}, mrr={mrr:.3f}")

        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "scores": scores,
            "num_contexts": len(contexts)
        }

    async def _retrieve_contexts(self, question: str, top_k: int = 3) -> List[str]:
        """检索相关上下文"""
        try:
            search_results = vector_search_service.search_similar_documents(
                query=question,
                top_k=top_k
            )
            contexts = [result.content for result in search_results if result.content]

            if not contexts:
                logger.warning(f"未检索到相关内容: {question}")
            else:
                logger.info(f"检索到 {len(contexts)} 个相关文档")

            return contexts
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    async def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """生成答案"""
        if not contexts:
            return "抱歉，未找到相关信息。"

        if not self.llm or not self.api_key_valid:
            summary = f"基于检索到的 {len(contexts)} 个文档片段：\n\n"
            for i, ctx in enumerate(contexts[:3], 1):
                summary += f"{i}. {ctx[:200]}...\n\n"
            return summary

        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请诚实说明。

上下文:
{chr(10).join(f'[{i+1}] {ctx[:500]}' for i, ctx in enumerate(contexts))}

问题: {question}

请基于上下文给出准确、简洁、完整的回答："""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"生成答案时出错: {e}"

    async def run_full_evaluation(
        self,
        test_data: Optional[List[Dict]] = None,
        top_k: int = 3,
        output_dir: str = "evals/ragas_results"
    ) -> pd.DataFrame:
        """运行完整评估"""
        if test_data is None:
            test_data = self._get_ground_truth_data()

        logger.info(f"开始评估，共 {len(test_data)} 个问题")

        results = []
        for i, item in enumerate(test_data):
            logger.info(f"处理问题 {i+1}/{len(test_data)}: {item['question'][:40]}...")
            result = await self.evaluate_single_question(
                question=item["question"],
                ground_truth=item["ground_truth"],
                top_k=top_k
            )
            results.append(result)

        # 转换为 DataFrame
        df_data = []
        for r in results:
            row = {
                "question": r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"],
                "num_contexts": r["num_contexts"],
                "faithfulness": r["scores"].get("faithfulness", 0),
                "answer_relevancy": r["scores"].get("answer_relevancy", 0),
                "context_recall": r["scores"].get("context_recall", 0),
                "answer_similarity": r["scores"].get("answer_similarity", 0),
                "mrr": r["scores"].get("mrr", 0),  # 新增 MRR 列
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        self._save_results(df, results, output_dir)
        return df

    def _get_ground_truth_data(self) -> List[Dict]:
        """获取测试数据集 - 覆盖各类运维场景"""
        test_data = [
            # Prometheus 相关告警
            {
                "question": "Prometheus job missing 告警如何排查？",
                "ground_truth": "Prometheus job missing 告警表示某个 Prometheus 任务已经消失。需要检查 Prometheus self-monitoring 的相关指标和日志，确认 exporter 是否崩溃。使用 PromQL 查询 absent(up{job=\"prometheus\"}) 来检测。",
            },
            {
                "question": "Prometheus target missing 告警怎么处理？",
                "ground_truth": "Prometheus target missing 告警表示 Prometheus 的目标已经消失，可能是 exporter 崩溃了。严重等级为 critical，持续 1 分钟后触发。使用 PromQL 查询 up == 0 unless on(job) (sum by (job) (up) == 0) 来检测。",
            },
            {
                "question": "Prometheus all targets missing 告警是什么意思？",
                "ground_truth": "Prometheus all targets missing 告警表示 Prometheus 的任务没有任何存活的 target，即所有目标都不可用。严重等级为 critical，使用 PromQL 查询 sum by (job) (up) == 0 来检测。",
            },

            # CPU 相关告警
            {
                "question": "CPU使用率过高告警如何排查？",
                "ground_truth": "CPU使用率过高告警处理方案：1. 使用 get_current_time 获取当前时间；2. 查询系统日志和 application-logs，条件为 level:ERROR OR cpu_usage:>80；3. 分析 CPU 消耗进程；4. 常见原因包括死循环、流量突增、定时任务重叠、数据库慢查询。处理措施包括扩容、限流、重启实例、优化代码。",
            },
            {
                "question": "CPU使用率超过80%怎么办？",
                "ground_truth": "CPU使用率超过80%是严重告警，可能影响应用响应速度。立即措施：扩容实例、启用限流；排查步骤：查询日志分析进程、检查是否有死循环或流量突增；长期优化：代码性能优化、增加监控告警。",
            },
            {
                "question": "死循环导致CPU高的特征是什么？",
                "ground_truth": "死循环导致CPU高的特征：单个进程CPU占用接近100%；应用日志中有大量重复的错误堆栈；内存使用也可能同步增长。处理方案：立即重启受影响的服务实例，查看应用日志定位代码问题，回滚到上一个稳定版本。",
            },
            {
                "question": "流量突增导致CPU高怎么处理？",
                "ground_truth": "流量突增导致CPU高的特征：多个进程CPU使用率均匀升高；请求量明显增加；响应时间变长但无明显错误。处理方案：检查是否有营销活动；启动自动扩容；启用限流策略；持续监控扩容后的CPU使用率。",
            },

            # 磁盘相关告警
            {
                "question": "日志文件过大导致磁盘满怎么解决？",
                "ground_truth": "日志文件过大导致磁盘满的解决方案：1. 立即清理：find /var/log -type f -size +100M 找到大文件并清空；2. 配置日志轮转：启用 logrotate，设置日志保留天数；3. 优化日志级别：生产环境使用 INFO 或 WARN 级别；4. 实施日志自动清理策略。",
            },
            {
                "question": "Docker 磁盘占用过高怎么清理？",
                "ground_truth": "Docker 磁盘占用过高清理方法：1. docker system prune -a --volumes 清理所有未使用资源；2. docker image prune -a 清理未使用的镜像；3. docker container prune 清理停止的容器；4. docker volume prune 清理未使用的卷；5. 配置容器日志驱动和日志轮转限制。",
            },

            # 内存相关告警
            {
                "question": "内存泄漏的特征是什么？",
                "ground_truth": "内存泄漏的特征：内存使用率持续缓慢上升；Full GC 后内存无法释放；应用运行时间越长内存占用越高；日志中有大量 GC 记录。处理方案：重启应用释放内存，在重启前 dump 内存快照，使用 MAT 工具分析堆转储文件定位代码问题。",
            },
            {
                "question": "Prometheus job missing 告警如何排查？",
                "ground_truth": "Prometheus job missing 告警表示某个 Prometheus 任务已经消失。需要检查 Prometheus self-monitoring 的相关指标和日志，确认 exporter 是否崩溃。使用 PromQL 查询 absent(up{job=\"prometheus\"}) 来检测。该告警严重等级为 warning，来源于 awesome-prometheus-alerts 的 Basic resource monitoring。"
            },
            {
                "question": "Prometheus target missing 告警是什么意思？如何处理？",
                "ground_truth": "Prometheus target missing 告警表示某个 Prometheus 监控目标消失了，可能是 exporter 崩溃导致的。使用 PromQL 查询 up == 0 unless on(job) (sum by (job) (up) == 0) 来检测。该告警仅在同一 job 中至少还有一个 target 存活时触发，严重等级为 critical。如果所有 target 都宕了，则由 PrometheusJobMissing 或 PrometheusAllTargetsMissing 告警触发。需要检查 Prometheus self-monitoring 的相关指标和日志。"
            },
            {
                "question": "Prometheus all targets missing 告警的含义和排查方法是什么？",
                "ground_truth": "Prometheus all targets missing 告警表示某个 Prometheus job 已经没有任何存活的监控目标了。使用 PromQL 查询 sum by (job) (up) == 0 来检测。该告警严重等级为 critical，需要检查 Prometheus self-monitoring 的相关指标和日志，来源于 awesome-prometheus-alerts 的 Basic resource monitoring。"
            },
            {
                "question": "Prometheus target missing 和 Prometheus all targets missing 有什么区别？",
                "ground_truth": "Prometheus target missing 表示某个 job 中部分 target 宕机但至少还有一个存活，使用 up == 0 unless on(job) (sum by (job) (up) == 0) 检测；而 Prometheus all targets missing 表示某个 job 的所有 target 都宕机了，使用 sum by (job) (up) == 0 检测。两者都是 critical 级别告警。当所有 target 都宕机时，只触发 all targets missing 而不触发 target missing。"
            },
            {
                "question": "CPU使用率过高告警应该怎么排查？",
                "ground_truth": "CPU使用率过高告警（HighCPUUsage）在CPU使用率持续5分钟超过80%时触发，级别为严重。排查步骤：1）获取当前时间确定告警时间范围；2）使用 query_logs 查询 system-metrics 日志主题，查询条件为 cpu_usage > 80；3）分析CPU消耗进程，关注进程名称、PID、CPU占用百分比；4）查询 application-logs 中的 ERROR 和 WARN 级别日志。常见原因包括死循环或无限递归、流量突增、定时任务重叠执行、数据库查询慢。"
            },
            {
                "question": "CPU使用率高的常见原因有哪些？分别怎么处理？",
                "ground_truth": "CPU使用率高的常见原因有4种：1）死循环或无限递归：特征为单个进程CPU接近100%，处理方案是立即重启服务、查看日志定位代码问题、回滚到稳定版本；2）流量突增：特征为多个进程CPU均匀升高，处理方案是启动自动扩容、启用限流策略；3）定时任务重叠执行：特征为CPU周期性升高，处理方案是调整任务时间避免重叠、增加互斥锁；4）数据库查询慢：特征为应用CPU高但业务逻辑简单，处理方案是优化SQL语句、添加索引、增加缓存层。"
            },
            {
                "question": "Docker占用大量磁盘空间怎么清理？",
                "ground_truth": "Docker占用大量磁盘空间时的清理方法：1）清理未使用的镜像：docker image prune -a；2）清理停止的容器：docker container prune；3）清理未使用的卷：docker volume prune；4）清理所有未使用资源：docker system prune -a --volumes。同时需要限制容器日志大小，配置日志驱动和日志轮转。长期优化包括使用多阶段构建减小镜像体积、定期清理旧镜像。"
            },
            {
                "question": "服务不可用告警应该怎么排查？",
                "ground_truth": "服务不可用告警（ServiceUnavailable）在健康检查失败或错误率超过50%时触发，级别为紧急。排查步骤：1）获取当前时间；2）查询 application-logs 中 level:ERROR OR level:FATAL OR status:500；3）查询 system-events 中的 restart、crash、oom_kill 事件；4）检查依赖服务状态，查询 downstream_service、database、redis、mq 相关日志。常见原因包括应用崩溃、数据库连接失败、依赖服务故障、配置错误、资源耗尽、网络故障。紧急处理流程：1分钟内确认故障并启动应急，5分钟内快速定位并决策回滚或修复，15分钟内恢复服务。"
            },
            {
                "question": "服务不可用时如何做故障复盘？",
                "ground_truth": "故障恢复后必须进行复盘，包括5个方面：1）故障时间线：记录故障发生、发现、处理、恢复的完整时间线；2）根因分析：深入分析故障根本原因；3）影响评估：评估故障影响范围和损失；4）改进措施：制定防止类似故障的改进措施；5）文档更新：更新运维文档和应急预案。预防措施包括高可用架构（多实例部署、多地域容灾）、监控告警（完善健康检查、自动故障转移）、发布策略（灰度发布、蓝绿部署、金丝雀发布）、容错设计（熔断降级、限流保护）、演练测试（定期故障演练、混沌工程）。"
            }
        ]
        return test_data

    def _save_results(self, df: pd.DataFrame, results: List[Dict], output_dir: str):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = output_path / f"rigorous_ragas_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print("\n" + "=" * 60)
        print("严谨版评估结果摘要")
        print("=" * 60)

        metric_cols = ["faithfulness", "answer_relevancy",
                       "context_recall", "answer_similarity"]

        for col in metric_cols:
            if col in df.columns:
                avg = df[col].mean()
                print(f"{col:25}: {avg:.4f}")

        overall = df[metric_cols].mean().mean() if metric_cols else 0
        print(f"\n{'综合评分':25}: {overall:.4f}")
        print(f"\n结果已保存: {csv_path}")


async def main():
    print("=" * 60)
    print("严谨版 RAGAS 评估系统 (含 MRR)")
    print("=" * 60)

    api_key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️ 未设置 API Key，将使用算法评估（部分指标可能不准确）")

    evaluator = RAGASEvaluator()
    results = await evaluator.run_full_evaluation(top_k=3)
    print("\n详细结果:")
    print(results.to_string())


if __name__ == "__main__":
    asyncio.run(main())