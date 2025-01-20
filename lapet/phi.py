import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

from .handler import ModelHandler

class Phi4ModelHandler():
    def load_model_and_tokenizer(self, device, model_id):
        # Configure quantization with CPU offloading enabled
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Create device map for CPU offloading
        device_map = {
            'model.embed_tokens': 0,
            'model.layers': 0,
            'model.norm': 0,
            'lm_head': 'cpu'  # Offload language model head to CPU
        }
        
        # Load model with quantization and device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        print(model_id + " loaded.")
        return tokenizer, model

    def post_process_output(self, prompt, output):
        # Remove the original prompt from the output
        output = output[len(prompt)-1:]
        
        # Clean up any chat markers and system text
        output = output.replace("<|im_start|>", "").replace("<|im_end|>", "")
        output = output.replace("system", "").replace("assistant", "").replace("user", "").strip()
        
        # Extract JSON response using the same pattern as Llama2ModelHandler
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        matches = re.findall(pattern, output)
        last_match = None
        if matches:
            last_match = matches[-1]
        return {last_match[0]: last_match[1]} if last_match else output 