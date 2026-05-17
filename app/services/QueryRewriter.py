from langchain_deepseek import ChatDeepSeek
from typing import List, Optional


class QueryRewriter:
    """查询改写器 - 使用 LLM 将用户问题改写为更适合检索的形式"""

    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm

    async def rewrite(
            self,
            query: str,
            history: Optional[List[dict]] = None,
            rewrite_type: str = "retrieval"
    ) -> str:
        """
        改写查询

        Args:
            query: 原始问题
            history: 对话历史（用于多轮对话）
            rewrite_type: 改写类型 (retrieval/hyde/multi_query)

        Returns:
            改写后的查询
        """

        if rewrite_type == "retrieval":
            return await self._rewrite_for_retrieval(query, history)
        elif rewrite_type == "hyde":
            return await self._hyde_rewrite(query)
        elif rewrite_type == "multi_query":
            return await self._multi_query_rewrite(query)
        else:
            return query

    async def _rewrite_for_retrieval(self, query: str, history: Optional[List[dict]] = None) -> str:
        """改写为适合检索的形式"""

        prompt = f"""你是一个查询优化专家。请将用户的原始问题改写为更适合知识库检索的形式。

        改写规则：
        1. 消除指代不清（"它"、"这个"等替换为具体名词）
        2. 补充隐含的上下文信息
        3. 使用专业、规范的术语
        4. 保持问题完整，不改变原意
        5. 输出只包含改写后的问题，不要有其他内容
        
        {f"对话历史：{history}" if history else ""}
        
        原始问题：{query}
        
        改写后的问题："""

        try:
            response = await self.llm.ainvoke(prompt)
            rewritten = response.content.strip()
            print(f"Query Rewrite: {query} -> {rewritten}")
            return rewritten
        except Exception as e:
            print(f"Query Rewrite 失败: {e}")
            return query

    async def _hyde_rewrite(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)
        先生成假设性答案，再用答案去检索
        """
        prompt = f"""请根据以下问题，生成一个假设性的答案。这个答案不需要完全准确，只需像真实文档一样。

        问题：{query}
        
        假设性答案："""

        try:
            response = await self.llm.ainvoke(prompt)
            hypothetical_answer = response.content.strip()
            print(f"HyDE: 生成假设答案长度 {len(hypothetical_answer)}")
            return hypothetical_answer
        except Exception as e:
            print(f"HyDE 失败: {e}")
            return query

    async def _multi_query_rewrite(self, query: str) -> List[str]:
        """生成多个角度的查询，提高召回率"""

        prompt = f"""请从不同角度将以下问题改写为 3 个不同的检索查询。

        原始问题：{query}
        
        要求：
        1. 每个查询关注问题的不同方面
        2. 保持查询简洁、规范
        3. 每行一个查询
        
        3个不同角度的查询："""

        try:
            response = await self.llm.ainvoke(prompt)
            lines = response.content.strip().split('\n')
            queries = [line.strip() for line in lines if line.strip()]
            print(f"Multi-Query: 生成 {len(queries)} 个变体")
            return queries if queries else [query]
        except Exception as e:
            print(f"Multi-Query 失败: {e}")
            return [query]