"""
Pipeline package for AI Avatar Tutor.
Contains modules for speech-to-text, RAG retrieval,
LLM response generation, avatar video creation, and orchestration.
"""
from pipeline.orchestrator import PipelineOrchestrator
__all__ = ["PipelineOrchestrator"]