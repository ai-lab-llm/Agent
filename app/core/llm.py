from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from app.config import settings

_llm_singleton = None


def get_chat_llm():
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton

    tok = AutoTokenizer.from_pretrained(settings.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    mdl = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype="auto",
        device_map={"": settings.device_map},  # "auto" or specific device
    )

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=settings.max_new_tokens,
        do_sample=False,
        temperature=settings.temperature,
        top_p=1.0,
        return_full_text=False,
    )
    base_llm = HuggingFacePipeline(pipeline=gen)
    _llm_singleton = ChatHuggingFace(llm=base_llm)
    return _llm_singleton