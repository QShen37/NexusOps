"""
RAGAS 评估对比脚本
对比以下两个 RAG pipeline 的性能：
  1. 基础 RAG（原始向量检索 vector_search_service）
  2. 增强 RAG（QueryRewriter + MultiRecallVectorSearchService）

用法：
    cd D:\\1 沈琪\\工作\\项目\\AI_ops\\app
    python services/run_ragas_evals.py
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ── 路径修复，确保 app.* 包可以被导入 ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # AI_ops/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
# ─────────────────────────────────────────────────────────────────

import pandas as pd
from loguru import logger
from langchain_deepseek import ChatDeepSeek

from app.config import config
from app.services.ragas_evals import RAGASEvaluator
from app.services.vector_search_service import vector_search_service
from app.services.QueryRewriter import QueryRewriter
from app.services.MultiRecallVectorSearchService import multi_recall_vector_search_service


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def load_test_data(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载测试数据（JSON 格式）。"""
    if filepath is None:
        filepath = str(Path(__file__).parent / "test_data_template.json")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"加载测试数据：{len(data)} 条，来自 {filepath}")
    return data


def _make_llm() -> ChatDeepSeek:
    """创建共享 LLM 实例。"""
    return ChatDeepSeek(
        model=config.DEEPSEEk_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
    )


def _print_scores(scores: Dict[str, float], title: str = ""):
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "answer_similarity", "mrr"]
    print(f"\n{'─' * 50}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 50}")
    for m in metrics:
        if m in scores:
            print(f"  {m:25s}: {scores[m]:.4f}")
    overall = sum(scores.get(m, 0) for m in metrics if m in scores)
    count = sum(1 for m in metrics if m in scores)
    if count:
        print(f"  {'综合评分':25s}: {overall / count:.4f}")
    print(f"{'─' * 50}")


# ═══════════════════════════════════════════════════════════════
# Pipeline 1：基础 RAG
# ═══════════════════════════════════════════════════════════════

class BasicRAGEvaluator(RAGASEvaluator):
    """
    继承 RigorousRAGASEvaluator，保持原始的检索逻辑不变：
      _retrieve_contexts → vector_search_service.search_similar_documents
      _generate_answer   → LLM 生成答案
    """

    async def _retrieve_contexts(self, question: str, top_k: int = 3) -> List[str]:
        """原始向量检索（来自父类，这里显式覆盖以便日志清晰）。"""
        try:
            results = vector_search_service.search_similar_documents(
                query=question,
                top_k=top_k,
            )
            contexts = [r.content for r in results if r.content]
            logger.info(f"[BasicRAG] 检索到 {len(contexts)} 个文档")
            return contexts
        except Exception as e:
            logger.error(f"[BasicRAG] 检索失败: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# Pipeline 2：QueryRewriter + MultiRecallVectorSearchService
# ═══════════════════════════════════════════════════════════════

class AdvancedRAGEvaluator(RAGASEvaluator):
    """
    在父类基础上替换检索阶段：
      1. QueryRewriter 对原始问题进行改写（retrieval 模式）
      2. MultiRecallVectorSearchService 进行多路召回
    答案生成和评估指标计算复用父类逻辑。
    """

    def __init__(self, api_key: Optional[str] = None, rewrite_type: str = "retrieval"):
        super().__init__(api_key=api_key)
        self.rewrite_type = rewrite_type

        llm = _make_llm()
        self.query_rewriter = QueryRewriter(llm=llm)
        self.multi_recall_service = multi_recall_vector_search_service

        # 为 Reranker 注入 LLM
        self.multi_recall_service.reranker.llm = llm
        logger.info(f"[AdvancedRAG] 初始化完成，改写模式: {rewrite_type}")

    async def _retrieve_contexts(self, question: str, top_k: int = 3) -> List[str]:
        """改写查询 → 多路召回。"""
        try:
            # Step 1: 改写查询
            rewritten = await self.query_rewriter.rewrite(
                query=question,
                rewrite_type=self.rewrite_type,
            )

            # multi_query 返回 List[str]，其余返回 str
            if isinstance(rewritten, list):
                queries = rewritten
            else:
                queries = [rewritten]

            logger.info(f"[AdvancedRAG] 改写查询: {question!r} -> {queries}")

            # Step 2: 多路召回（对每个改写后的查询）
            seen_ids: set = set()
            all_contexts: List[str] = []

            for q in queries:
                results = await self.multi_recall_service.search_similar_documents(
                    query=q,
                    top_k=top_k,
                    recall_type="rerank",  # 混合检索 + 重排序
                )
                for r in results:
                    if r.id not in seen_ids and r.content:
                        seen_ids.add(r.id)
                        all_contexts.append(r.content)

            # 截取 top_k 条
            contexts = all_contexts[:top_k]
            logger.info(f"[AdvancedRAG] 多路召回后得到 {len(contexts)} 个文档（去重后）")
            return contexts

        except Exception as e:
            logger.error(f"[AdvancedRAG] 检索失败: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# 评估执行器
# ═══════════════════════════════════════════════════════════════

async def run_pipeline(
    evaluator: RAGASEvaluator,
    test_data: List[Dict[str, Any]],
    top_k: int = 3,
) -> pd.DataFrame:
    """对给定 pipeline 执行完整评估，返回 DataFrame。"""
    results = []
    for i, item in enumerate(test_data):
        logger.info(f"评估 [{i+1}/{len(test_data)}]: {item['question'][:50]}...")
        result = await evaluator.evaluate_single_question(
            question=item["question"],
            ground_truth=item["ground_truth"],
            top_k=top_k,
        )
        results.append(result)

    # 整理成 DataFrame
    rows = []
    for r in results:
        row = {
            "question": r["question"],
            "answer": r["answer"][:200] + "..." if len(r["answer"]) > 200 else r["answer"],
            "num_contexts": r["num_contexts"],
        }
        row.update(r["scores"])
        rows.append(row)

    return pd.DataFrame(rows)


def avg_scores(df: pd.DataFrame) -> Dict[str, float]:
    """从 DataFrame 计算各指标均值。"""
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "answer_similarity", "mrr"]
    return {m: float(df[m].mean()) for m in metrics if m in df.columns}


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          RAGAS 评估系统 - RAG Pipeline 性能对比                ║
║                                                              ║
║  Pipeline 1: 基础 RAG（原始向量检索）                          ║
║  Pipeline 2: 增强 RAG（QueryRewriter + 多路召回 + 重排序）      ║
╚══════════════════════════════════════════════════════════════╝
""")

    # 加载测试数据
    test_data = load_test_data()

    # 用户可通过命令行参数指定 top_k
    top_k = 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent.parent / "evals" / "compare_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_results: Dict[str, Dict] = {}

    # ── Pipeline 1: 基础 RAG ─────────────────────────────────
    print("\n[1/2] 运行基础 RAG 评估...")
    basic_eval = BasicRAGEvaluator()
    basic_df = await run_pipeline(basic_eval, test_data, top_k=top_k)
    basic_scores = avg_scores(basic_df)
    pipeline_results["basic_rag"] = {"scores": basic_scores, "df": basic_df}
    _print_scores(basic_scores, "基础 RAG（原始向量检索）")

    # 保存详细结果
    basic_csv = output_dir / f"basic_rag_{timestamp}.csv"
    basic_df.to_csv(basic_csv, index=False, encoding="utf-8")
    logger.info(f"基础 RAG 结果已保存: {basic_csv}")

    # ── Pipeline 2: 增强 RAG ─────────────────────────────────
    print("\n[2/2] 运行增强 RAG 评估（QueryRewriter + 多路召回）...")
    advanced_eval = AdvancedRAGEvaluator(rewrite_type="retrieval")
    advanced_df = await run_pipeline(advanced_eval, test_data, top_k=top_k)
    advanced_scores = avg_scores(advanced_df)
    pipeline_results["advanced_rag"] = {"scores": advanced_scores, "df": advanced_df}
    _print_scores(advanced_scores, "增强 RAG（QueryRewriter + 多路召回 + 重排序）")

    # 保存详细结果
    advanced_csv = output_dir / f"advanced_rag_{timestamp}.csv"
    advanced_df.to_csv(advanced_csv, index=False, encoding="utf-8")
    logger.info(f"增强 RAG 结果已保存: {advanced_csv}")

    # ── 对比汇总 ─────────────────────────────────────────────
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "answer_similarity", "mrr"]
    print(f"\n{'═' * 70}")
    print("  📊 评估对比汇总")
    print(f"{'═' * 70}")
    print(f"  {'指标':25s}  {'基础 RAG':>10}  {'增强 RAG':>10}  {'提升':>8}")
    print(f"  {'─' * 57}")
    for m in metrics:
        b = basic_scores.get(m, 0)
        a = advanced_scores.get(m, 0)
        diff = a - b
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "─")
        print(f"  {m:25s}  {b:>10.4f}  {a:>10.4f}  {arrow} {abs(diff):.4f}")
    print(f"{'═' * 70}")

    # 保存汇总 JSON
    summary = {
        "timestamp": timestamp,
        "top_k": top_k,
        "num_questions": len(test_data),
        "pipelines": {
            "basic_rag": basic_scores,
            "advanced_rag": advanced_scores,
        },
    }
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 汇总结果已保存: {summary_path}")
    print(f"  📁 基础 RAG 详情: {basic_csv}")
    print(f"  📁 增强 RAG 详情: {advanced_csv}\n")


if __name__ == "__main__":
    asyncio.run(main())
