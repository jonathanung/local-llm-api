from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import torch

load_dotenv()

device = os.getenv("DEVICE")
text_generation_model_name = os.getenv("TEXT_GENERATION_MODEL_NAME")

tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)
model = AutoModelForCausalLM.from_pretrained(text_generation_model_name).to(device)

async def text_generation_pipeline(prompt, return_full_text, max_new_tokens, temperature, repetition_penalty, top_k, top_p, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if not return_full_text:
        generated_text = generated_text[len(prompt):]
    
    return [{"generated_text": generated_text}]