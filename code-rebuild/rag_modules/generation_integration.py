"""
生成集成模块 - 考研招生信息RAG系统
负责将检索到的招生信息上下文与大语言模型结合，生成准确、专业、友好的回答
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ============================================================================
# 异常类定义
# ============================================================================

class GenerationError(Exception):
    """生成异常"""
    pass


class LLMConnectionError(Exception):
    """LLM连接异常"""
    pass


class ContextTooLongError(Exception):
    """上下文过长异常"""
    pass


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class GenerationConfig:
    """生成模块配置"""

    # 模型配置
    model_name: str = "kimi-k2-0711-preview"
    temperature: float = 0.1
    max_tokens: int = 2048

    # 上下文配置
    max_context_length: int = 6000

    # 查询处理配置
    enable_query_rewrite: bool = True
    enable_query_router: bool = True


# ============================================================================
# 主类
# ============================================================================

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""

    def __init__(
        self,
        model_name: str = "kimi-k2-0711-preview",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        config: Optional[GenerationConfig] = None
    ):
        """
        初始化生成集成模块

        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
            config: 可选配置对象
        """
        if config is not None:
            self.config = config
        else:
            self.config = GenerationConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

        self.model_name = self.config.model_name
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        try:
            self.llm = MoonshotChat(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                moonshot_api_key=api_key
            )
            logger.info("LLM初始化完成")
        except Exception as e:
            raise LLMConnectionError(f"LLM连接失败: {e}")

    # ========================================================================
    # 查询预处理
    # ========================================================================

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 将模糊、口语化的查询转化为更精确的检索查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        if not self.config.enable_query_rewrite:
            return query

        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的考研信息查询，判断是否需要重写以提高检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含完整学校名称：如"北京大学招生简章"、"清华大学专业目录"
   - 明确的专业询问：如"计算机科学与技术复试线"、"电子信息专硕招生人数"
   - 具体的政策查询：如"2024年考研报名时间"、"推免生申请条件"

2. **模糊不清的查询**（需要重写）：
   - 学校简称：如"北大"、"清华"、"浙大"
   - 过于宽泛：如"考研"、"研究生"、"怎么考"
   - 缺乏具体信息：如"分数线"、"招生人数"、"学费"

重写原则：
- 保持原意不变
- 补充完整的学校名称
- 增加考研相关术语
- 保持简洁性

示例：
- "北大计算机" → "北京大学计算机科学与技术专业招生信息"
- "分数线" → "考研复试分数线录取线"
- "怎么考北大" → "北京大学报考条件报考流程"
- "北京大学招生简章" → "北京大学招生简章"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询意图分发到不同的回答生成策略

        Args:
            query: 用户查询

        Returns:
            路由类型 ('simple', 'list', 'multi_info', 'detail', 'general')
        """
        if not self.config.enable_query_router:
            return 'general'

        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下五种类型之一：

1. 'simple' - 用户想要获取单个简单事实信息，只需一两句话回答
   例如：计算机专业招多少人、分数线是多少、学费多少

2. 'list' - 用户想要获取学校/专业名称列表或推荐，只需要名称
   例如：推荐几个985计算机、有哪些好考的211、北京的学校有哪些

3. 'multi_info' - 用户想要获取多个项目的具体信息对比或汇总
   例如：所有专业的招生人数分别是多少、各专业考试科目对比、各个学校分数线汇总

4. 'detail' - 用户想要单个项目的详细政策、报考条件等
   例如：北京大学招生简章内容、复试怎么准备、报考条件是什么

5. 'general' - 其他一般性问题或概念解释
   例如：什么是推免、考研什么时候报名、学硕和专硕的区别

请只返回分类结果：simple、list、multi_info、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        if result in ['simple', 'list', 'multi_info', 'detail', 'general']:
            return result
        else:
            return 'general'

    # ========================================================================
    # 上下文构建
    # ========================================================================

    def _build_context(self, docs: List[Document], max_length: int = None) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关招生信息。"

        if max_length is None:
            max_length = self.config.max_context_length

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【招生信息 {i}】"
            if 'school' in doc.metadata:
                metadata_info += f" {doc.metadata['school']}"
            if 'college' in doc.metadata and doc.metadata.get('college') and doc.metadata['college'] != 'null':
                metadata_info += f" | {doc.metadata['college']}"
            if 'major' in doc.metadata and doc.metadata.get('major') and doc.metadata['major'] != 'null':
                metadata_info += f" | {doc.metadata['major']}"
            if 'year' in doc.metadata and doc.metadata.get('year'):
                metadata_info += f" | {doc.metadata['year']}年"
            if 'info_type' in doc.metadata:
                metadata_info += f" | {doc.metadata['info_type']}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + "="*50 + "\n".join(context_parts)

    # ========================================================================
    # 回答生成
    # ========================================================================

    def generate_simple_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成简洁回答 - 适用于简单事实查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            简洁的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的考研咨询顾问。请根据招生信息，简洁地回答考生的问题。

用户问题: {question}

相关招生信息:
{context}

回答要求：
- 直接回答用户的问题，不要展开说明
- 回答控制在1-3句话
- 如果信息不足，直接说"暂无相关信息"
- 不要添加额外解释或建议

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_multi_info_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成多项目信息汇总回答 - 适用于批量信息提取查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            信息汇总回答
        """
        context = self._build_context(context_docs, max_length=8000)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的考研咨询顾问。请根据招生信息，为考生汇总多个项目的信息。

用户问题: {question}

相关招生信息:
{context}

回答要求：
1. 使用表格或列表形式，清晰展示每个项目的信息
2. 确保列出所有提到的项目，不要遗漏
3. 每个项目单独一行，格式统一
4. 如果某些项目缺少特定信息，标注"未说明"
5. 最后可以简要总结关键信息

示例格式：
| 专业名称 | 招生人数 | 考试科目 |
|---------|---------|---------|
| 计算机科学与技术 | 9人 | 408计算机学科专业基础 |
| 软件工程 | 5人 | 408计算机学科专业基础 |
| ... | ... | ... |

或列表格式：
1. 计算机科学与技术：招生9人
2. 软件工程：招生5人
...

请根据用户问题选择最合适的格式回答：

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答 - 适用于通用类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的考研咨询顾问。请根据以下招生信息回答考生的问题。

用户问题: {question}

相关招生信息:
{context}

请提供准确、实用的回答。注意：
- 信息来源于官方渠道，请准确传达
- 如果信息不足，请诚实说明
- 涉及年份的信息请明确标注
- 重要信息可用加粗强调

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_detail_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成详细结构化回答 - 适用于政策解读类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            详细的结构化回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的考研咨询顾问。请根据招生信息，为考生提供详细的解答。

用户问题: {question}

相关招生信息:
{context}

请按以下结构组织回答（根据实际内容调整）：

## 📋 基本信息
[学校/学院/专业的基本介绍]

## 📝 招生计划
[招生人数、推免比例等]

## 🎯 报考条件
[学历要求、专业限制等]

## 📚 考试科目
[初试科目、复试内容等]

## 📊 历年数据
[分数线、报录比等（如有）]

## ⚠️ 注意事项
[重要提醒、截止日期等]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 信息准确优先，来源不清的请标注
- 时间敏感信息请特别提醒

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的招生信息。"

        # 提取学校/专业信息
        items = []
        seen = set()

        for doc in context_docs:
            school = doc.metadata.get('school', '未知学校')
            major = doc.metadata.get('major', '')
            year = doc.metadata.get('year', '')

            # 构建唯一标识
            if major and major != 'null':
                key = f"{school}_{major}"
            else:
                key = school

            if key not in seen:
                seen.add(key)
                items.append({
                    'school': school,
                    'major': major if major and major != 'null' else '',
                    'info_type': doc.metadata.get('info_type', ''),
                    'year': year
                })

        # 构建回答
        if len(items) == 1:
            item = items[0]
            result = f"为您找到：{item['school']}"
            if item['major']:
                result += f" - {item['major']}"
            if item['year']:
                result += f" ({item['year']}年)"
        elif len(items) <= 5:
            result = "为您推荐以下院校/专业：\n"
            for i, item in enumerate(items, 1):
                line = f"{i}. {item['school']}"
                if item['major']:
                    line += f" - {item['major']}"
                if item['year']:
                    line += f" ({item['year']}年)"
                result += line + "\n"
        else:
            result = "为您推荐以下院校/专业：\n"
            for i, item in enumerate(items[:5], 1):
                line = f"{i}. {item['school']}"
                if item['major']:
                    line += f" - {item['major']}"
                if item['year']:
                    line += f" ({item['year']}年)"
                result += line + "\n"
            result += f"\n还有其他 {len(items)-5} 个选择可供参考。"

        return result

    # ========================================================================
    # 流式输出
    # ========================================================================

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的考研咨询顾问。请根据以下招生信息回答考生的问题。

用户问题: {question}

相关招生信息:
{context}

请提供准确、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_detail_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的考研咨询顾问。请根据招生信息，为考生提供详细的解答。

用户问题: {question}

相关招生信息:
{context}

请按以下结构组织回答：
## 📋 基本信息
## 📝 招生计划
## 🎯 报考条件
## 📚 考试科目
## ⚠️ 注意事项

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    # ========================================================================
    # 主入口
    # ========================================================================

    def generate(
        self,
        query: str,
        context_docs: List[Document],
        answer_type: str = "auto"
    ) -> str:
        """
        生成回答主入口

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            answer_type: 回答类型 ('auto', 'simple', 'list', 'multi_info', 'detail', 'general')

        Returns:
            生成的回答
        """
        if answer_type == "auto":
            answer_type = self.query_router(query)

        if answer_type == 'simple':
            return self.generate_simple_answer(query, context_docs)
        elif answer_type == 'list':
            return self.generate_list_answer(query, context_docs)
        elif answer_type == 'multi_info':
            return self.generate_multi_info_answer(query, context_docs)
        elif answer_type == 'detail':
            return self.generate_detail_answer(query, context_docs)
        else:
            return self.generate_basic_answer(query, context_docs)

    # ========================================================================
    # 兼容性方法（保留旧方法名）
    # ========================================================================

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成详细回答（兼容旧方法名）"""
        return self.generate_detail_answer(query, context_docs)

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """生成详细回答 - 流式输出（兼容旧方法名）"""
        return self.generate_detail_answer_stream(query, context_docs)


# ============================================================================
# 便捷函数
# ============================================================================

def generate_answer(
    query: str,
    vectorstore,
    chunks: List[Document],
    answer_type: str = "auto",
    top_k: int = 5
) -> str:
    """
    一站式生成回答

    Args:
        query: 用户查询
        vectorstore: 向量存储
        chunks: 文档块列表
        answer_type: 回答类型 ('auto', 'basic', 'detail', 'list')
        top_k: 检索数量

    Returns:
        生成的回答
    """
    from .retrieval_optimization import RetrievalOptimizationModule

    # 初始化模块
    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    generation_module = GenerationIntegrationModule()

    # 检索
    docs = retrieval_module.hybrid_search(query, top_k=top_k)

    # 生成回答
    return generation_module.generate(query, docs, answer_type)
