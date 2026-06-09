# -*- coding: utf-8 -*-
import os, json, time, uuid
from typing import Optional, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")
LORA_PATH = os.path.join(os.path.dirname(__file__), "lora_output", "best")
LORA_PATH = ""
HOST = "0.0.0.0"
PORT = 8000

app = FastAPI(title="Qwen3.5-0.8B OpenAI-compatible API")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3.5-0.8b"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map=0, trust_remote_code=True, torch_dtype=torch.bfloat16,
)
model.eval()

if os.path.exists(LORA_PATH):
    print(f"Loading LoRA adapter from {LORA_PATH}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_PATH)

print(f"Model loaded on {model.device}")


def build_prompt(messages: List[ChatMessage]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"{msg.content}\n"
        elif msg.role == "user":
            prompt += f"问题：{msg.content}\n\n回答："
        elif msg.role == "assistant":
            prompt += msg.content
    return prompt


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = build_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": generated}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": out.shape[1] - inputs["input_ids"].shape[1],
            "total_tokens": out.shape[1],
        },
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "qwen3.5-0.8b", "object": "model", "created": int(time.time()), "owned_by": "user"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print(f"Server: http://{HOST}:{PORT}/v1/chat/completions")
    uvicorn.run(app, host=HOST, port=PORT)