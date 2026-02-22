

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Re-import BitsAndBytesConfig

#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
#MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None

def get_llm_model_hf():
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        print("Loading model once...")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32  # CPU-safe
        )

        _model.eval()

    return _model, _tokenizer
