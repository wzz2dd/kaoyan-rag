"""
RAG系统配置文件
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from rag_modules.data_preparation import LoaderConfig


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/kaoyan"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 数据加载配置
    loader_config: LoaderConfig = field(default_factory=LoaderConfig)

    def __post_init__(self):
        """初始化后的处理"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'loader_config': self.loader_config,
        }


DEFAULT_CONFIG = RAGConfig()