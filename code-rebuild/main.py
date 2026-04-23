"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataLoaderModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaoyanRAGSystem:
    """考研信息RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        print("初始化数据加载模块...")
        self.data_module = DataLoaderModule(self.config.data_path, config=self.config.loader_config)

        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            print("加载招生信息文档...")
            self.data_module.load_documents()
            self.data_module.standardize_metadata()
            self.data_module.identify_info_units()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
            if self.config.loader_config.build_hierarchy:
                self.data_module.build_relationships()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            print("加载招生信息文档...")
            self.data_module.load_documents()
            self.data_module.standardize_metadata()
            self.data_module.identify_info_units()

            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            print("保存向量索引...")
            self.index_module.save_index()

            if self.config.loader_config.build_hierarchy:
                self.data_module.build_relationships()

        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   学校数量: {stats['schools']}")
        print(f"   学院数量: {stats['colleges']}")
        print(f"   专业数量: {stats['majors']}")

        print("✅ 知识库构建完成！")

    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        if route_type == 'list':
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                major = chunk.metadata.get('major', '未知专业')
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{major}({section_title})")
                else:
                    chunk_info.append(f"{major}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        if not relevant_chunks:
            return "抱歉，没有找到相关的招生信息。请尝试其他关键词。"

        if route_type == 'list':
            print("📋 生成信息列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            return self.generation_module.generate_list_answer(question, relevant_docs)

        print("获取完整文档...")
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        print("✍️ 生成详细回答...")

        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
            return self.generation_module.generate_step_by_step_answer(question, relevant_docs)

        if stream:
            return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
        return self.generation_module.generate_basic_answer(question, relevant_docs)

    def _extract_filters_from_query(self, query: str) -> dict:
        filters = {}
        for degree in DataLoaderModule.get_supported_degree_types():
            if degree in query:
                filters['degree_type'] = degree
                break

        for mode in DataLoaderModule.get_supported_study_modes():
            if mode in query:
                filters['study_mode'] = mode
                break

        for info_type in DataLoaderModule.get_supported_info_types():
            if info_type in query:
                filters['info_type'] = info_type
                break

        year_match = re.search(r"(20\d{2})", query)
        if year_match:
            filters['year'] = int(year_match.group(1))

        return filters

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🎓 考研信息RAG系统 - 交互式问答 🎓")
        print("=" * 60)
        print("💡 解决你的考研信息查询难题！")

        self.initialize_system()
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用考研信息RAG系统！")


def main():
    """主函数"""
    try:
        rag_system = KaoyanRAGSystem()
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()