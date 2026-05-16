"""embedding_study — 임베딩 LLM 학습 패키지.

임베딩 모델을 활용한 텍스트 벡터화, 의미 검색, RAG 파이프라인 예제.
"""

from src.embedding_study._interface import Embedder, VectorStore
from src.embedding_study.basics import SentenceEmbedder, demonstrate_embedding_basics
from src.embedding_study.ollama_embedder import OllamaEmbedder
from src.embedding_study.search import (
    InMemoryVectorStore,
    build_knowledge_base,
    demonstrate_semantic_search,
)
from src.embedding_study.rag import RAGPipeline, demonstrate_rag_pipeline
from src.embedding_study.llm_integration import (
    DeepSeekLLM,
    ScenarioResult,
    build_knowledge_store,
    scenario_1_llm_only,
    scenario_2_llm_rag_keyword,
    scenario_3_embedding_llm,
    scenario_4_full_rag,
    run_all_scenarios,
)

__all__ = [
    "Embedder",
    "VectorStore",
    "SentenceEmbedder",
    "OllamaEmbedder",
    "InMemoryVectorStore",
    "RAGPipeline",
    "build_knowledge_base",
    "demonstrate_embedding_basics",
    "demonstrate_semantic_search",
    "demonstrate_rag_pipeline",
    "DeepSeekLLM",
    "ScenarioResult",
    "build_knowledge_store",
    "scenario_1_llm_only",
    "scenario_2_llm_rag_keyword",
    "scenario_3_embedding_llm",
    "scenario_4_full_rag",
    "run_all_scenarios",
]
