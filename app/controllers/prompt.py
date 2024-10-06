from fastapi import HTTPException
from app.utils.model import text_generation_pipeline

async def process_prompt(prompt: str, return_full_text: bool = False, max_new_tokens: int = 200, temperature: float = 0.7, repetition_penalty: float = 1.1, top_k: int = 50, top_p: float = 0.95, **kwargs):
    try:
        result = await text_generation_pipeline(prompt, return_full_text, max_new_tokens, temperature, repetition_penalty, top_k, top_p, **kwargs)
        return [{"generated_text": result[0]["generated_text"]}]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))