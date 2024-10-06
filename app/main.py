from fastapi import FastAPI, Depends, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware as CORS
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from app.controllers import prompt as prompt_controller

load_dotenv()

app = FastAPI()

print(os.getenv("CORS_ORIGINS"))
# CORS
origins = os.getenv("CORS_ORIGINS", "").split(",")
app.add_middleware(
    CORS,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define a Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str
    return_full_text: bool = False
    max_new_tokens: int = 200
    temperature: float = 0.7
    repetition_penalty: float = 1.1
    top_k: int = 50
    top_p: float = 0.95

@app.post("/prompt")
async def prompt_endpoint(request: PromptRequest = Body(...)):
    return await prompt_controller.process_prompt(
        prompt=request.prompt,
        return_full_text=request.return_full_text,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        top_k=request.top_k,
        top_p=request.top_p
    )
