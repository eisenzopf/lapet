import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

class ModelHandler:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.system_prompt = config["system_prompt"]
        self.prompts = config["prompts"]
        self.models = config["models"]
        self.samples = config["samples"]
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.max_length = config["max_length"]
        self.dataset = self.load_dataset(config["dataset"])

    def generate_output(self, text):
        """Generates an output for a given input"""
        messages = [
            {"role": "system", "content": self.system_prompt },
            {"role": "user", "content": text },
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        outputs = ""
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        responses = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prompt, responses

    def load_dataset(self, dataset):
        """Loads an external CSV dataset via URL and prepares a dataframe for storing the output"""
        print("Loading dataset...")
        df = pd.read_csv(dataset)
        df = pd.DataFrame(df)
        conversation_ids = df['conversation_id'].unique()
        sampled_ids = np.random.choice(conversation_ids, size=self.samples, replace=False)
        conversation_texts = {}
        for id in sampled_ids:
          filtered_df = df[df['conversation_id'] == id]
          text_accumulator = ""
          for _, row in filtered_df.iterrows():
            text_accumulator += f"{row['speaker']}: {row['text']}\n"
          conversation_texts[id] = text_accumulator.strip()
        return conversation_texts

    def load_model_and_tokenizer(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(self.device)
        print(model_id + " loaded.")
        return tokenizer, model

    def post_process_output(self, prompt, output):
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        matches = re.findall(pattern, output)
        last_match = None
        if matches:
            last_match = matches[-1]
        return {last_match[0]: last_match[1]} if last_match else output
    
    def prepare_output(self):
        rows = []
        column_names = ['id', 'model']  # start with 'id' and 'model' columns
        for prompt in self.prompts:
            column_names.append(prompt["name"] + '.input')
            column_names.append(prompt["name"] + '.output')
        df = pd.DataFrame(columns=column_names)

        for model_name in self.models.items():
            for data_id, data_value in self.dataset.items():  
                row = {'id': data_id, 'model': model_name}
                for prompt in self.prompts:
                    input_column_name = prompt["name"] + '.input'
                    output_column_name = prompt["name"] + '.output'
                    row[input_column_name] = f"### Conversation ###\n{data_value}\n\n### Instruction ###\n{prompt['prompt']}\n\n### Output ###\n" 
                    row[output_column_name] = ""
                rows.append(row)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        return df

    def process_dataset(self):
        df = self.prepare_output()
        for model_name, model_handler in self.models.items():
            self.current_model = model_name
            print(f"Loading {model_name}...")
            handler = model_handler()
            self.tokenizer, self.model = handler.load_model_and_tokenizer(self.device, model_name)
            for index, row in df[df['model'] == model_name].iterrows(): 
                print(f"Generating outputs for {row['id']}")
                for col in df.columns:
                    if col.endswith('.input'):
                        output_col = col.replace('.input', '.output')
                        prompt, output = self.generate_output(row[col])
                        output = self.post_process_output(prompt, output)
                        df.at[index, output_col] = output
            self.unload_model(model_name)
        return df

    def unload_model(self, model_id):
        del self.model, self.tokenizer
        import torch
        torch.cuda.empty_cache()
        print(f"{model_id} unloaded from memory.")