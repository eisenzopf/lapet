import itertools
import pandas as pd
import numpy as np
from openai import OpenAI
import random
import re

class LLMJudge:
    def __init__(self, config, df):
        self.config = config
        self.dataset = self.prepare_dataset(df)

    def extract_output_by_name(self, value):
        if isinstance(value, float) and np.isnan(value):
            return ''
        pattern = re.compile(r'\{\s*["\'](.+?)["\']\s*:\s*["\'](.+?)["\']\s*\}')
        matches = re.findall(pattern, value)
        return matches[-1][1].strip() if matches else value

    def evaluate(self):
        client = OpenAI(
            organization=self.config['judge']['organization'],
            project=self.config['judge']['project'],
            api_key=self.config['judge']['api_key']
        )

        for index, row in self.dataset.iterrows():
            flip = random.randint(0, 1)
            comp1_model, comp1_value, comp2_model, comp2_value = (
                (row['model1'], row['comp1.value'], row['model2'], row['comp2.value']) if flip == 0 else
                (row['model2'], row['comp2.value'], row['model1'], row['comp1.value'])
            )

            entries = f"""Ok here is the instruction that we provided to both participants:
Welcome to our customer service analysis tool. You will be provided with transcripts of conversations between customers and service agents. Your task is to follow the instruction and output a response from each conversation. Focus on providing concise outputs that could be useful for follow-up actions and ensure that your outputs are directly relevant to the discussed topics. This prompt is meant to ensure that you understand the essence of the customer's concerns and can articulate it succinctly in a structured format that is easy for both human and machine processing. Continue with this approach for the upcoming conversations.

{row['instruction']}

Ok, here are the answers from the two participants for {row['test.name']}:
participant 1: {comp1_model}: {comp1_value}
participant 2: {comp2_model}: {comp2_value}

Pick the participant's response that you prefer the most and explain why. Please format your output in YAML like:
preference: <1 | 2 | tie >
explanation: <explain the reasons for your choice>
"""
            messages = [
                {"role": "system", "content": "You are going to pick an answer from two different participants based on an instruction. You should pick the entry that follows instructions the best."},
                {"role": "user", "content": entries}
            ]
            completion = client.chat.completions.create(model="gpt-4o", messages=messages)
            completion_content = completion.choices[0].message.content

            preference, explanation = self.parse_completion(completion_content)

            if preference:
                self.dataset.at[index, 'preference'] = (
                    comp1_model if "1" in preference else
                    comp2_model if "2" in preference else
                    "tie"
                )
                self.dataset.at[index, 'explanation'] = explanation
                print(f"preference: {preference}\nexplanation: {explanation}")
                print("-----------------")

    def parse_completion(self, completion_content):
        preference, explanation = None, None
        for line in completion_content.split('\n'):
            if line.startswith('preference:'):
                preference = line.split('preference:')[1].strip()
            elif line.startswith('explanation:'):
                explanation = line.split('explanation:')[1].strip()
        return preference, explanation

    def prepare_dataset(self, df):
        output_columns = [col for col in df.columns if col.endswith('.output')]
        comparison_rows = []

        for id_, group in df.groupby('id'):
            models = group['model'].unique()
            model_pairs = itertools.combinations(models, 2)

            for model1, model2 in model_pairs:
                for col in output_columns:
                    row1, row2 = group[group['model'] == model1], group[group['model'] == model2]
                    if not row1.empty and not row2.empty:
                        test_name = col.replace('.output', '')
                        comp1_value = self.extract_output_by_name(row1[col].values[0])
                        comp2_value = self.extract_output_by_name(row2[col].values[0])
                        instruction_value = row1[f'{test_name}.input'].values[0]
                        comparison_rows.append([id_, instruction_value, model1, model2, test_name, comp1_value, comp2_value, None, None])

        comparison_df = pd.DataFrame(comparison_rows, columns=['id', 'instruction', 'model1', 'model2', 'test.name', 'comp1.value', 'comp2.value', 'preference', 'explanation'])
        return comparison_df
