import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

class ModelHandler:
    def __init__(self, model_id, config):
        self.model_id = model_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model = self.load_model_and_tokenizer()
        self.system_prompt = config["system_prompt"]
        self.prompts = config["prompts"]
        self.dataset = self.load_dataset(config["dataset"])
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.max_length = config["max_length"]

    def load_model_and_tokenizer(self):
        """Loads model and tokenizer, should be overridden by subclasses to specify model type."""
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_output(self, texts):
        """Generates responses from the Llama model for given texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

    def load_dataset(self, dataset):
        """Loads an external CSV dataset via URL fetch"""
        df = pd.read_csv(dataset)
        df = pd.DataFrame(df)
        conversation_ids = df['conversation_id'].unique()
        sampled_ids = np.random.choice(conversation_ids, size=20, replace=False)
        conversation_texts = {}
        for id in sampled_ids:
          filtered_df = df[df['conversation_id'] == id]
          text_accumulator = ""
          for _, row in filtered_df.iterrows():
            text_accumulator += f"{row['speaker']}: {row['text']}\n"
          conversation_texts[id] = text_accumulator.strip()
        return conversation_texts

    def load_model_and_tokenizer(self):
        """Specific loader for Llama model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        terminators = [
          tokenizer.eos_token_id,
          tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        model = AutoModelForCausalLM.from_pretrained(self.model_id, eos_token_id=terminators)
        model.to(self.device)
        return tokenizer, model

    def unload_model(self):
        # Logic to unload model from memory
        del self.model, self.tokenizer
        import torch
        torch.cuda.empty_cache()
        print(f"{self.model_id} unloaded from memory")

    def post_process_output(self, output):
        """Extracts and returns content based on the predefined pattern from generated output."""
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        match = re.search(pattern, output)
        return {match.group(1): match.group(2)} if match else output