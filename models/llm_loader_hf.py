

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Re-import BitsAndBytesConfig

#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME="Qwen-2.5 0.5B instruct"
#MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None

def get_llm_model_hf():
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        print("Loading model...")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Define BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config, # Pass the quantization config here
        )

    return _model, _tokenizer
