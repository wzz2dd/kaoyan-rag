"""
考研招生信息RAG系统主程序
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

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
    """考研招生信息RAG系统主类"""

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

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("正在初始化考研RAG系统...")

        # 1. 初始化数据加载模块
        print("初始化数据加载模块...")
        self.data_module = DataLoaderModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载招生文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载招生文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   学校数量: {stats['schools']}")
        print(f"   学院数量: {stats['colleges']}")
        print(f"   专业数量: {stats['majors']}")
        print(f"   平均块大小: {stats['avg_chunk_size']:.0f} 字符")

        print("知识库构建完成！")

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

        print(f"\n用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块（自动应用元数据过滤）
        print("检索相关文档...")
        filters = self._extract_filters_from_query(question)

        # 根据查询类型调整检索数量
        search_top_k = self.config.top_k
        if route_type == 'list':
            search_top_k = max(self.config.top_k * 2, 20)  # 列表查询返回更多结果
        elif route_type == 'multi_info':
            search_top_k = max(self.config.top_k * 3, 30)  # 多信息查询返回更多结果

        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=search_top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=search_top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                school = chunk.metadata.get('school', '未知学校')
                major = chunk.metadata.get('major', '未知专业')
                info_type = chunk.metadata.get('info_type', '未知类型')
                # 构建显示信息
                if major and major != 'null':
                    chunk_info.append(f"{school}-{major}({info_type})")
                else:
                    chunk_info.append(f"{school}({info_type})")

            # 显示所有或部分结果
            display_count = min(len(chunk_info), 10)
            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info[:display_count])}")
            if len(chunk_info) > display_count:
                print(f"   ... 还有 {len(chunk_info) - display_count} 个文档块")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的招生信息。请尝试其他学校、专业名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回列表
            print("生成信息列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                school = doc.metadata.get('school', '未知学校')
                major = doc.metadata.get('major', '')
                if major and major != 'null':
                    doc_names.append(f"{school}-{major}")
                else:
                    doc_names.append(school)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        elif route_type == 'simple':
            # 简单查询：简洁回答
            print("生成简洁回答...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                school = doc.metadata.get('school', '未知学校')
                major = doc.metadata.get('major', '')
                info_type = doc.metadata.get('info_type', '')
                if major and major != 'null':
                    doc_names.append(f"{school}-{major}({info_type})")
                else:
                    doc_names.append(f"{school}({info_type})")

            if doc_names:
                print(f"找到文档: {', '.join(doc_names[:5])}")

            return self.generation_module.generate_simple_answer(question, relevant_docs)
        elif route_type == 'multi_info':
            # 多信息查询：汇总多个项目的信息
            print("生成信息汇总...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                school = doc.metadata.get('school', '未知学校')
                major = doc.metadata.get('major', '')
                info_type = doc.metadata.get('info_type', '')
                if major and major != 'null':
                    doc_names.append(f"{school}-{major}({info_type})")
                else:
                    doc_names.append(f"{school}({info_type})")

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_multi_info_answer(question, relevant_docs)
        else:
            # 详细/通用查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                school = doc.metadata.get('school', '未知学校')
                major = doc.metadata.get('major', '')
                info_type = doc.metadata.get('info_type', '')
                if major and major != 'null':
                    doc_names.append(f"{school}-{major}({info_type})")
                else:
                    doc_names.append(f"{school}({info_type})")

            if doc_names:
                print(f"找到文档: {', '.join(doc_names[:5])}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        从用户问题中提取元数据过滤条件

        Args:
            query: 用户查询

        Returns:
            过滤条件字典
        """
        filters = {}

        # 1. 提取学校名称
        schools = DataLoaderModule.get_supported_schools() if hasattr(DataLoaderModule, 'get_supported_schools') else []
        for school in schools:
            if school in query:
                filters['school'] = school
                break

        # 2. 提取年份 (如 2024, 2025, 2026)
        year_match = re.search(r'\b(20[12]\d)\b', query)
        if year_match:
            filters['year'] = int(year_match.group(1))

        # 3. 提取学位类型
        degree_keywords = {
            '学术学位': ['学硕', '学术学位', '学术型', '学术硕士'],
            '专业学位': ['专硕', '专业学位', '专业型', '专业硕士']
        }
        for degree_type, keywords in degree_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    filters['degree_type'] = degree_type
                    break
            if 'degree_type' in filters:
                break

        # 4. 提取学习方式
        study_keywords = {
            '全日制': ['全日制'],
            '非全日制': ['非全日制', '非全', '在职']
        }
        for mode, keywords in study_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    filters['study_mode'] = mode
                    break
            if 'study_mode' in filters:
                break

        # 5. 提取信息类型
        info_types = DataLoaderModule.get_supported_info_types()
        info_type_keywords = {
            '招生简章': ['招生简章', '简章', '招生说明'],
            '专业目录': ['专业目录', '招生专业', '专业一览'],
            '复试方案': ['复试方案', '复试办法', '复试安排', '复试通知'],
            '调剂信息': ['调剂', '接收调剂'],
            '录取情况': ['录取', '拟录取', '录取名单'],
        }
        for info_type, keywords in info_type_keywords.items():
            if info_type in info_types:
                for keyword in keywords:
                    if keyword in query:
                        filters['info_type'] = info_type
                        break
            if 'info_type' in filters:
                break

        return filters

    def search_by_school(self, school: str, query: str = "") -> List[str]:
        """
        按学校搜索招生信息

        Args:
            school: 学校名称
            query: 可选的额外查询条件

        Returns:
            专业/信息列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        # 使用元数据过滤搜索
        search_query = query if query else school
        filters = {"school": school}

        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)

        # 提取专业名称
        results = []
        for doc in docs:
            major = doc.metadata.get('major', '')
            info_type = doc.metadata.get('info_type', '')
            if major and major != 'null':
                result = f"{major}({info_type})"
            else:
                result = f"{info_type}"
            if result not in results:
                results.append(result)

        return results

    def search_by_major(self, major: str, school: str = None) -> List[str]:
        """
        按专业搜索招生信息

        Args:
            major: 专业名称
            school: 可选的学校筛选

        Returns:
            招生信息列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        filters = {}
        if school:
            filters['school'] = school

        docs = self.retrieval_module.metadata_filtered_search(major, filters, top_k=5)

        # 提取信息
        results = []
        for doc in docs:
            school_name = doc.metadata.get('school', '未知学校')
            major_name = doc.metadata.get('major', '未知专业')
            info_type = doc.metadata.get('info_type', '未知类型')
            year = doc.metadata.get('year', '')
            results.append(f"{school_name} {year}年 {major_name} - {info_type}")

        return results

    def get_admission_info(self, school: str, major: str = None) -> str:
        """
        获取指定学校/专业的招生信息

        Args:
            school: 学校名称
            major: 专业名称（可选）

        Returns:
            招生信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 构建查询
        query = f"{school} {major} 招生信息" if major else f"{school} 招生信息"
        filters = {"school": school}

        # 搜索相关文档
        docs = self.retrieval_module.metadata_filtered_search(query, filters, top_k=5)

        # 生成回答
        question = f"{school}{' '+major if major else ''}的招生信息有哪些？"
        answer = self.generation_module.generate_basic_answer(question, docs)

        return answer

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("  考研招生信息RAG系统 - 交互式问答")
        print("=" * 60)
        print("支持查询: 招生简章、专业目录、复试方案、调剂信息等")
        print("示例问题:")
        print("  - 浙江大学计算机专业2026年招生情况")
        print("  - 有哪些学校招电子信息专硕？")
        print("  - 复试流程是什么？")
        print()

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用考研招生信息RAG系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = KaoyanRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
