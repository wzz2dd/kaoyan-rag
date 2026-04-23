"""
数据加载模块
"""

import logging
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)

try:
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML not installed
    yaml = None


@dataclass
class LoaderConfig:
    """加载器配置"""

    file_patterns: List[str] = field(default_factory=lambda: ["*.md"])
    exclude_patterns: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=lambda: ["school", "year"])
    custom_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    chunk_overlap: int = 50
    build_hierarchy: bool = True
    resolve_conflicts: bool = True
    merge_duplicates: bool = True
    output_metadata: bool = True
    log_level: str = "INFO"


class MetadataStandardizer:
    """元数据标准化器"""

    SCHOOL_MAPPING = {
        "北大": "北京大学",
        "清华": "清华大学",
        "人大": "中国人民大学",
    }

    DEGREE_TYPE_MAPPING = {
        "学硕": "学术学位",
        "专硕": "专业学位",
        "学术型": "学术学位",
        "专业型": "专业学位",
    }

    STUDY_MODE_MAPPING = {
        "全日制": "全日制",
        "非全": "非全日制",
        "在职": "非全日制",
    }

    EXAM_TYPE_MAPPING = {
        "统考": "全国统考",
        "全国统考": "全国统考",
        "推荐免试": "推荐免试",
    }

    def __init__(self, required_fields: List[str], custom_mappings: Optional[Dict[str, Dict[str, str]]] = None):
        self.required_fields = required_fields
        self.custom_mappings = custom_mappings or {}

    def standardize(self, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        errors = []

        school_mapping = {**self.SCHOOL_MAPPING, **self.custom_mappings.get("school", {})}
        degree_mapping = {**self.DEGREE_TYPE_MAPPING, **self.custom_mappings.get("degree_type", {})}
        study_mapping = {**self.STUDY_MODE_MAPPING, **self.custom_mappings.get("study_mode", {})}
        exam_mapping = {**self.EXAM_TYPE_MAPPING, **self.custom_mappings.get("exam_type", {})}

        if metadata.get("school") in school_mapping:
            metadata["school"] = school_mapping[metadata["school"]]
        if metadata.get("degree_type") in degree_mapping:
            metadata["degree_type"] = degree_mapping[metadata["degree_type"]]
        if metadata.get("study_mode") in study_mapping:
            metadata["study_mode"] = study_mapping[metadata["study_mode"]]
        if metadata.get("exam_type") in exam_mapping:
            metadata["exam_type"] = exam_mapping[metadata["exam_type"]]

        year_value = metadata.get("year")
        if isinstance(year_value, str) and year_value.isdigit():
            metadata["year"] = int(year_value)

        for field in self.required_fields:
            if not metadata.get(field):
                errors.append(f"必填字段 {field} 缺失或为空")

        return metadata, errors


class ContentCleaner:
    """内容清洗器"""

    FILTER_PATTERNS = [
        r"<!--.*?-->",
        r"\[//\]: # .*",
        r"^> \[!NOTE\].*$",
    ]

    DOWNGRADE_PATTERNS = [
        r"^> ",
        r"^\*{2}提示\*{2}",
    ]

    def clean(self, content: str) -> str:
        cleaned = content
        for pattern in self.FILTER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)
        return cleaned.strip()


class RelationshipBuilder:
    """关系构建器"""

    def build_hierarchy(self, documents: List[Document]) -> Dict[str, Any]:
        hierarchy = {"schools": {}, "statistics": {}}

        for doc in documents:
            metadata = doc.metadata
            school = metadata.get("school", "未知学校")
            college = metadata.get("college", "未知学院")
            major = metadata.get("major", "未知专业")

            school_bucket = hierarchy["schools"].setdefault(school, {"name": school, "colleges": {}})
            college_bucket = school_bucket["colleges"].setdefault(college, {"name": college, "majors": {}})
            major_bucket = college_bucket["majors"].setdefault(major, {"name": major, "doc_ids": [], "info_types": []})

            major_bucket["doc_ids"].append(metadata.get("doc_id"))
            info_type = metadata.get("info_type")
            if info_type and info_type not in major_bucket["info_types"]:
                major_bucket["info_types"].append(info_type)

        hierarchy["statistics"] = {
            "total_schools": len(hierarchy["schools"]),
            "total_colleges": sum(len(school["colleges"]) for school in hierarchy["schools"].values()),
            "total_majors": sum(len(college["majors"]) for school in hierarchy["schools"].values() for college in school["colleges"].values()),
            "total_docs": len(documents),
        }
        return hierarchy


class DataLoaderModule:
    """数据加载模块 - 负责数据加载、标准化、切块和关系构建"""

    SOURCE_PRIORITY = {
        "官网": 100,
        "研招网": 90,
        "教育部": 95,
        "论坛": 50,
        "经验分享": 40,
        "其他": 30,
    }

    INFO_TYPES = [
        "招生简章",
        "专业目录",
        "考试大纲",
        "复试方案",
        "调剂信息",
        "录取情况",
        "导师信息",
        "历年真题",
        "经验分享",
        "招生计划",
        "分数线",
        "报录比",
        "其他",
    ]

    DEGREE_TYPES = ["学术学位", "专业学位"]
    STUDY_MODES = ["全日制", "非全日制"]

    def __init__(self, data_path: str, config: Optional[LoaderConfig | Dict[str, Any]] = None):
        self.data_path = data_path
        self.config = config if isinstance(config, LoaderConfig) else LoaderConfig(**(config or {}))
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.hierarchy: Dict[str, Any] = {}

        self.standardizer = MetadataStandardizer(self.config.required_fields, self.config.custom_mappings)
        self.cleaner = ContentCleaner()
        self.relationship_builder = RelationshipBuilder()

    @classmethod
    def get_supported_degree_types(cls) -> List[str]:
        return cls.DEGREE_TYPES

    @classmethod
    def get_supported_study_modes(cls) -> List[str]:
        return cls.STUDY_MODES

    @classmethod
    def get_supported_info_types(cls) -> List[str]:
        return cls.INFO_TYPES

    def load_documents(self) -> List[Document]:
        logger.info(f"正在从 {self.data_path} 加载文档...")

        documents: List[Document] = []
        data_path_obj = Path(self.data_path)

        include_files = []
        for pattern in self.config.file_patterns:
            include_files.extend(data_path_obj.rglob(pattern))

        for md_file in include_files:
            if any(md_file.match(pat) for pat in self.config.exclude_patterns):
                continue
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata, body = self._parse_front_matter(content)

                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = md_file.resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = md_file.as_posix()
                doc_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                metadata.update(
                    {
                        "doc_id": doc_id,
                        "source": str(md_file),
                        "doc_type": "parent",
                    }
                )

                doc = Document(page_content=body, metadata=metadata)
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _parse_front_matter(self, content: str) -> Tuple[Dict[str, Any], str]:
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        front_matter = parts[1]
        body = parts[2].lstrip()

        if yaml:
            try:
                parsed = yaml.safe_load(front_matter) or {}
                return parsed, body
            except Exception:
                logger.warning("YAML头部解析失败，回退到简单解析")

        metadata = {}
        for line in front_matter.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return metadata, body

    def _enhance_metadata(self, doc: Document):
        metadata = doc.metadata
        file_path = Path(metadata.get("source", ""))

        if not metadata.get("school"):
            metadata["school"] = file_path.parts[0] if file_path.parts else "未知学校"
        metadata.setdefault("college", "未知学院")
        metadata.setdefault("major", file_path.stem)

        info_type = metadata.get("info_type")
        if not info_type:
            for info in self.INFO_TYPES:
                if info in doc.page_content:
                    info_type = info
                    break
        metadata["info_type"] = info_type or "其他"

        if metadata.get("data_source"):
            metadata["source"] = metadata.get("data_source")
        metadata["source_priority"] = self.SOURCE_PRIORITY.get(metadata.get("source", "其他"), 30)

    def standardize_metadata(self) -> List[Document]:
        standardized = []
        for doc in self.documents:
            updated_metadata, errors = self.standardizer.standardize(doc.metadata)
            doc.metadata = updated_metadata
            if errors:
                doc.metadata["metadata_errors"] = errors
                logger.warning(f"元数据校验失败: {doc.metadata.get('source')} - {errors}")
            standardized.append(doc)
        return standardized

    def identify_info_units(self) -> List[Document]:
        for doc in self.documents:
            doc.page_content = self.cleaner.clean(doc.page_content)
        return self.documents

    def chunk_documents(self) -> List[Document]:
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        chunks = self._markdown_header_split()

        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        headers_to_split_on = [
            ("#", "level_1"),
            ("##", "level_2"),
            ("###", "level_3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        all_chunks: List[Document] = []

        for doc in self.documents:
            try:
                md_chunks = markdown_splitter.split_text(doc.page_content)

                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('major', '未知')} 未能按标题分割，可能缺少标题结构")

                parent_id = doc.metadata.get("doc_id")

                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update(
                        {
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "doc_type": "child",
                            "chunk_index": i,
                        }
                    )
                    self.parent_child_map[child_id] = parent_id

                expanded_chunks = []
                for chunk in md_chunks:
                    if len(chunk.page_content) > self.config.max_chunk_size:
                        expanded_chunks.extend(size_splitter.split_documents([chunk]))
                    else:
                        expanded_chunks.append(chunk)

                all_chunks.extend(expanded_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def build_relationships(self) -> Dict[str, Any]:
        self.hierarchy = self.relationship_builder.build_hierarchy(self.documents)
        return self.hierarchy

    def process(self) -> Tuple[List[Document], List[Document], Dict[str, Any]]:
        self.load_documents()
        self.standardize_metadata()
        self.identify_info_units()
        self.chunk_documents()
        if self.config.build_hierarchy:
            self.build_relationships()
        return self.documents, self.chunks, self.hierarchy

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        schools = set()
        colleges = set()
        majors = set()

        for doc in self.documents:
            schools.add(doc.metadata.get("school", "未知学校"))
            colleges.add(doc.metadata.get("college", "未知学院"))
            majors.add(doc.metadata.get("major", "未知专业"))

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "schools": len(schools),
            "colleges": len(colleges),
            "majors": len(majors),
            "avg_chunk_size": sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0,
        }

    def export_metadata(self, output_path: str):
        import json

        metadata_list = []
        for doc in self.documents:
            metadata_list.append(
                {
                    "source": doc.metadata.get("source"),
                    "school": doc.metadata.get("school"),
                    "college": doc.metadata.get("college"),
                    "major": doc.metadata.get("major"),
                    "year": doc.metadata.get("year"),
                    "info_type": doc.metadata.get("info_type"),
                    "data_source": doc.metadata.get("data_source"),
                    "content_length": len(doc.page_content),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_relevance: Dict[str, int] = {}
        parent_docs_map: Dict[str, Document] = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("doc_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        sorted_parent_ids = sorted(parent_relevance.keys(), key=lambda x: parent_relevance[x], reverse=True)

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        parent_info = []
        for doc in parent_docs:
            major_name = doc.metadata.get("major", "未知专业")
            parent_id = doc.metadata.get("doc_id")
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{major_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs


# Backward compatibility alias
DataPreparationModule = DataLoaderModule