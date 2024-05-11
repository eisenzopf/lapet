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
        #self.tokenizer, self.model = self.load_model_and_tokenizer()
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
        #self.results = self.prepare_output()

    def generate_output(self, text):
        """Generates an output for a given input"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        responses = ' '.join(responses)
        return responses

    def process_dataset(self):
        df = self.prepare_output()
        for index, row in df.iterrows():
            for col in df.columns:
                if col.endswith('.input'):
                    output_col = col.replace('.input', '.output')
                    df.at[index, output_col] = self.generate_output(row[col])
        return df

    def load_dataset(self, dataset):
        """Loads an external CSV dataset via URL and prepares a dataframe for storing the output"""
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
    
    def prepare_input(self):
        """Prepares the model input for inference based on """
        print()
    
    def prepare_output(self):
        """Create a DataFrame with the required columns based on the list of dictionaries"""
        rows = []
        column_names = ['id', 'model']  # start with 'id' and 'model' columns
        for prompt in self.prompts:
            column_names.append(prompt["name"] + '.input')
            column_names.append(prompt["name"] + '.output')
        df = pd.DataFrame(columns=column_names)

        for model_name, model_handler in self.models.items():
            for data_id, text in self.dataset.items():
                row = {'id': data_id, 'model': model_name}
                for prompt in self.prompts:
                    input_column_name = prompt["name"] + '.input'
                    output_column_name = prompt["name"] + '.output'
                    # Construct the input value by combining system prompt and column-specific text
                    row[input_column_name] = f"{self.system_prompt}\n\n{self.dataset[data_id]}\n\n{prompt['prompt']}"
                    # Placeholder for outputs as examples
                    row[output_column_name] = ""
                rows.append(row)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        #df.to_csv('data_output.csv', index=False)
        return df

    def post_process_output(self, output):
        """Extracts and returns content based on the predefined pattern from generated output."""
        pattern = re.compile(r'\{\s*"(.+?)"\s*:\s*"(.+?)"\s*\}')
        match = re.search(pattern, output)
        return {match.group(1): match.group(2)} if match else output

    def unload_model(self):
        # Logic to unload model from memory
        del self.model, self.tokenizer
        import torch
        torch.cuda.empty_cache()
        print(f"{self.model_id} unloaded from memory")