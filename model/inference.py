import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import build_messages

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# Loaded ONCE at startup, not per request
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": os.getenv("APP_VERSION", "dev")}

@app.post("/chat")
def chat(req: ChatRequest):
    text = tokenizer.apply_chat_template(
        build_messages(req.prompt), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens,
                             do_sample=True, temperature=0.7)
    reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"reply": reply}