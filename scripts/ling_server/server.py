#!/usr/bin/env python3
"""
FastAPI server wrapping HuggingFace transformers for Ling-3.0-tiny-int4.

Exposes OpenAI-compatible /v1/chat/completions so the existing
pilot_llm_v*.py CachedChatClient-compatible clients can call Ling
without vLLM or SGLang.

Endpoint:  http://localhost:8000/v1/chat/completions
Model id:  Ling-3.0-tiny
GPU:       CUDA_VISIBLE_DEVICES=3 (overridable)
Cache:     content-addressed at cache_dir/{sha256(endpoint + payload)}.json
            (matches the existing CachedChatClient cache key scheme)

Use:
    # in another shell, or background:
    python -m scripts.ling_server.server

    # then:
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Ling-3.0-tiny","messages":[{"role":"user","content":"hi"}], "max_tokens":16}'

Or via the bundled client (scripts.ling_client.LingClient):
    from scripts.ling_client import LingClient
    client = LingClient(endpoint="http://localhost:8000", cache_dir=Path("..."))
    out = client.call([{"role":"user","content":"hi"}], seed=20_260_902)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer

# --- paths ----------------------------------------------------------------- #

ROOT = Path("/storage/gaoym/sp500-forecastability-lab")
LING_MODEL_DIR = Path("/storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4")
DEFAULT_GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "3"))
DEFAULT_PORT = int(os.environ.get("LING_PORT", "8000"))
DEFAULT_CACHE_DIR = ROOT / "results/pilot_llm_v8/cache"

MAX_COMPLETION_TOKENS = 160  # matches V5 §3
REQUEST_TIMEOUT = 60.0  # matches V5 §3


# --- request / response models (OpenAI-compatible) ----------------------- #

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = MAX_COMPLETION_TOKENS
    top_p: float = 0.95
    seed: int | None = None  # V5 §3 passes _agent_seed(agent_index)
    stream: bool = False
    # We do not implement logprobs, tools, n>1, response_format, etc.


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# --- server state --------------------------------------------------------- #

app = FastAPI(title="Ling FastAPI Server", version="0.1")
_state: dict[str, Any] = {}


def _load_model(model_dir: Path, gpu: int) -> tuple[Any, Any]:
    """Load Ling tokenizer + model on the specified GPU.

    Returns (tokenizer, model). Uses trust_remote_code=True to pick up
    inclusionAI's custom BailingMoeV3 architecture in the model dir.
    """
    print(f"[server] loading tokenizer from {model_dir} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    # Ling ships its chat template as chat_template.jinja, not in
    # tokenizer_config.json. Load it explicitly.
    chat_template_path = model_dir / "chat_template.jinja"
    if chat_template_path.exists():
        tok.chat_template = chat_template_path.read_text(encoding="utf-8")
        print(f"[server] loaded chat_template from {chat_template_path}",
              flush=True)

    print(f"[server] patching BailingMoeV3 _init_weights to skip quantized layers...",
          flush=True)
    _patch_init_weights()

    print(f"[server] loading model on cuda:{gpu} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu}",
    )
    model.eval()
    print(f"[server] model loaded; dtype={next(model.parameters()).dtype}",
          flush=True)
    return tok, model


def _patch_init_weights() -> None:
    """Replace BailingMoeV3PreTrainedModel._init_weights to skip quantized
    Linear/Embedding layers (compressed-tensors int4 layers have no .weight
    attribute). Without this, AutoModel.from_pretrained fails when
    _initialize_weights calls _init_weights on a quantized layer.

    Patches the class in transformers_modules cache directly. The patch
    survives subsequent loads because transformers re-uses the cached
    module file (unless it's invalidated).
    """
    from pathlib import Path
    cache_root = Path.home() / ".cache/huggingface/modules/transformers_modules"
    target_files = list(cache_root.rglob("modeling_bailing_moe_v3.py"))
    if not target_files:
        print(f"[server] WARN: no cached modeling file found under {cache_root}",
              flush=True)
        return
    old = """    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()"""
    new = """    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if not hasattr(module, 'weight'):
                return  # PATCH: skip quantized Linear (no .weight attribute)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if not hasattr(module, 'weight'):
                return  # PATCH: skip quantized Embedding
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()"""
    for f in target_files:
        text = f.read_text(encoding="utf-8")
        if old in text:
            f.write_text(text.replace(old, new))
            print(f"[server] patched _init_weights in {f}", flush=True)
        else:
            # Already patched or different version
            if "PATCH: skip quantized Linear" in text:
                print(f"[server] _init_weights already patched in {f}", flush=True)
            else:
                print(f"[server] WARN: old _init_weights pattern not found in {f}",
                      flush=True)


def _cache_key(endpoint: str, payload: dict[str, Any]) -> str:
    """Content-addressed cache key, matches CachedChatClient's scheme."""
    material = {"endpoint": endpoint, "request": payload}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _write_cache(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- endpoints --------------------------------------------------------- ---

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "Ling-3.0-tiny",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "inclusionAI",
            "root": str(LING_MODEL_DIR),
            "max_model_len": 32768,
        }],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_state.get("model"))}


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest):
    if not _state.get("model"):
        raise HTTPException(status_code=503, detail="model not loaded")

    tok = _state["tokenizer"]
    model = _state["model"]
    cache_dir: Path = _state["cache_dir"]
    endpoint = _state["endpoint"]

    # Build canonical payload (matches CachedChatClient format)
    canonical_payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
    }
    cache_key = _cache_key(endpoint, canonical_payload)

    # Cache hit?
    cached = _read_cache(cache_dir, cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    # Build prompt
    messages = [m.model_dump() for m in req.messages]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc_inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(enc_inputs.input_ids.shape[1])

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc_inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,                  # temperature=0.0 → greedy
            temperature=None,                 # suppress generate's default warning
            top_p=None,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    seq = out.sequences[0]
    new_tokens = seq[enc_inputs.input_ids.shape[1]:]
    reply = tok.decode(new_tokens, skip_special_tokens=True)
    elapsed = time.time() - t0

    completion_tokens = int(new_tokens.shape[0])
    response: dict[str, Any] = {
        "id": f"chatcmpl-{cache_key[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        # extra metadata (not in OpenAI spec but useful for debugging)
        "_elapsed_seconds": elapsed,
        "_cache_key": cache_key,
    }

    _write_cache(cache_dir, cache_key, response)
    return response


# --- CLI / main --------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                   help=f"CUDA device index (default: {DEFAULT_GPU})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"port to bind (default: {DEFAULT_PORT})")
    p.add_argument("--host", default="0.0.0.0", help="bind host")
    p.add_argument("--model-dir", type=Path, default=LING_MODEL_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    tok, model = _load_model(args.model_dir, args.gpu)
    _state["tokenizer"] = tok
    _state["model"] = model
    _state["cache_dir"] = args.cache_dir
    _state["endpoint"] = f"http://localhost:{args.port}/v1/chat/completions"

    print(f"[server] starting uvicorn on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()