import itertools
import pandas as pd
import numpy as np
from openai import OpenAI

class Judge:
    def __init__(self, config):
        """initialize the judge class"""

    def pairwise_evaluation(self, df):
        """conducts a pairwise comparison using GPT-4o"""

    def prepare_dataset(self, df):
        # Get columns ending with '.output'
        output_columns = [col for col in df.columns if col.endswith('.output')]
        
        # Initialize a list to store rows for the new DataFrame
        comparison_rows = []

        # Group by 'id'
        for id_, group in df.groupby('id'):
            # Generate all pairwise combinations of models
            models = group['model'].unique()
            model_pairs = list(itertools.combinations(models, 2))
            
            for model1, model2 in model_pairs:
                for col in output_columns:
                    row1 = group[group['model'] == model1]
                    row2 = group[group['model'] == model2]
                    if not row1.empty and not row2.empty:
                        comp1_name = col.replace('.output', '')
                        comp1_value = row1[col].values[0]
                        comp2_name = col.replace('.output', '')
                        comp2_value = row2[col].values[0]
                        comparison_rows.append([id_, model1, model2, comp1_name, comp1_value, comp2_name, comp2_value, None])  # None for preference initially

        # Create the new DataFrame from the list of rows
        comparison_df = pd.DataFrame(comparison_rows, columns=['id', 'model1', 'model2', 'comp1.name', 'comp1.value', 'comp2.name', 'comp2.value', 'preference'])
        
        return comparison_df