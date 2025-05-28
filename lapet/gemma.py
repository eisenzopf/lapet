import re
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

from .handler import ModelHandler

class GemmaModelHandler():
    def load_model_and_tokenizer(self, device, model_id):
        # Enable TensorFloat32 for better performance
        torch.set_float32_matmul_precision('high')
        
        # Configure dynamo to prevent cache limit errors
        torch._dynamo.config.cache_size_limit = 64  # Increase cache size
        torch._dynamo.config.suppress_errors = True  # Suppress dynamo errors
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()
        
        print(model_id + " loaded.")
        return tokenizer, model

    def post_process_output(self, prompt, output):
        # Remove the original prompt from the output
        output = output[len(prompt)-1:]
        
        # Clean up any special tokens and markers
        output = output.replace("<bos>", "").replace("<eos>", "").strip()
        
        # Extract JSON response using the same pattern as other handlers
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        matches = re.findall(pattern, output)
        last_match = None
        if matches:
            last_match = matches[-1]
        return {last_match[0]: last_match[1]} if last_match else output 