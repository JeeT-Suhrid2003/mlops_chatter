"""FastAPI inference server.

On startup it downloads the LoRA adapter from the Hugging Face Hub (pinned to
MODEL_REVISION), loads the base model on CPU, merges the adapter, and serves /chat.
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = os.environ["MODEL_REPO"]                    # e.g. youruser/chat-model-lora
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")  # commit SHA, tag, or branch
HF_TOKEN = os.environ.get("HF_TOKEN")                    # needed for private repos
API_KEY = os.environ.get("API_KEY")                      # optional shared secret
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
MAX_NEW_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "256"))

state = {}
gen_lock = threading.Lock()  # CPU generation is serialized; extra requests wait their turn


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    path = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, token=HF_TOKEN)
    meta = {}
    meta_file = os.path.join(path, "metrics.json")
    if os.path.exists(meta_file):
        meta = json.load(open(meta_file))
    base = os.environ.get("BASE_MODEL") or meta.get("base_model") or "Qwen/Qwen2.5-0.5B-Instruct"

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, token=HF_TOKEN)
    model = PeftModel.from_pretrained(model, path).merge_and_unload().eval()

    state.update(
        tok=tok,
        model=model,
        info={
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "version": meta.get("version"),
            "eval_loss": meta.get("eval_loss"),
            "base_model": base,
            "load_seconds": round(time.time() - t0, 1),
        },
    )
    yield
    state.clear()


app = FastAPI(title="Chat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(min_length=1, max_length=20)
    max_new_tokens: int = Field(default=200, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ok"}


@app.get("/version")
def version():
    return state.get("info", {})


@app.post("/chat", dependencies=[Depends(check_key)])
def chat(req: ChatRequest):
    tok, model = state["tok"], state["model"]
    prompt = tok.apply_chat_template(
        [m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1536)
    sample = req.temperature > 0
    kwargs = {"do_sample": True, "temperature": req.temperature, "top_p": 0.9} if sample else {"do_sample": False}
    with gen_lock, torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=min(req.max_new_tokens, MAX_NEW_TOKENS_CAP),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            **kwargs,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"reply": text.strip(), "model_version": state["info"].get("version")}
