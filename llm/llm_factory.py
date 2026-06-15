from __future__ import annotations
import os
from typing import Union
from llm.llm_mock import MockLLM
from llm.llm_real import HeuristicPolicyGenerator
from llm.prompts import ProductionLLM

LLMBackend = Union[MockLLM, HeuristicPolicyGenerator, ProductionLLM]

def get_llm_by_name(name: str) -> LLMBackend:
    if name == "heuristic":
        return HeuristicPolicyGenerator()
    if name in ("real", "env"):
        return ProductionLLM()   # ← production prompts
    return MockLLM()

def get_llm() -> LLMBackend:
    use_real = os.environ.get("USE_REAL_LLM", "false").lower() in ("1", "true", "yes")
    if not use_real:
        return MockLLM()
    has_key = any([os.environ.get("OPENROUTER_API_KEY"), os.environ.get("OPENAI_API_KEY")])
    if not has_key:
        return MockLLM()
    return ProductionLLM()
