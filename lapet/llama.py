import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers

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
        # Create pipeline with bfloat16 precision and auto device mapping
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            trust_remote_code=True
        )
        
        # Extract tokenizer and model from pipeline
        tokenizer = pipeline.tokenizer
        model = pipeline.model
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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