import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .handler import ModelHandler
class Llama3ModelHandler():
      def load_model_and_tokenizer(self, device, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        terminators = [
          tokenizer.eos_token_id,
          tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        model = AutoModelForCausalLM.from_pretrained(model_id, eos_token_id=terminators)
        model.to(device)
        print(model_id + " loaded.")
        return tokenizer, model
      
from .handler import ModelHandler
class Llama2ModelHandler():
      def load_model_and_tokenizer(self, device, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device)
        print(model_id + " loaded.")
        return tokenizer, model
      
      def post_process_output(self, prompt, output):
        output = output[len(prompt)-1:]
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        matches = re.findall(pattern, output)
        last_match = None
        if matches:
            last_match = matches[-1]
        return {last_match[0]: last_match[1]} if last_match else output

class Llama31ModelHandler():
    def load_model_and_tokenizer(self, device, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Configure the proper RoPE scaling for Llama 3.1/3.2
        model_config = {
            "rope_scaling": {
                "type": "dynamic",
                "factor": 2.0
            }
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            config_overrides=model_config
        )
        
        print(model_id + " loaded.")
        return tokenizer, model
        
    def post_process_output(self, prompt, output):
        output = output[len(prompt)-1:]
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        matches = re.findall(pattern, output)
        last_match = None
        if matches:
            last_match = matches[-1]
        return {last_match[0]: last_match[1]} if last_match else output