import pandas as pd
import itertools
import numpy as np

# Sample DataFrame creation for demonstration
"""data = {
    'id': ['5e8231e4507d4b20aa4e1937e3560735', '5e8231e4507d4b20aa4e1937e3560735', '2983aa2c3a79434d861dfd10e5318dc0', '2983aa2c3a79434d861dfd10e5318dc0'],
    'model': ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct'],
    'intent-sentence.input': [
        'The following is a conversation between a customer and a support agent...',
        'The following is a conversation between a customer and a support agent...',
        'The following is a conversation between a customer and a support agent...',
        'The following is a conversation between a customer and a support agent...'
    ],
    'intent-sentence.output': [
        '\ngoal: The customer wants to find a better plan...',
        '\ngoal: The customer wants to find a better plan...',
        '\ngoal: Cancel the mobile hotspot plan.',
        '\ngoal: Cancel the mobile hotspot plan.'
    ],
    'intent-label.input': [
        'The following is a conversation between a customer and a support agent...',
        'The following is a conversation between a customer and a support agent...',
        'The following is a conversation between a customer and a support agent...',
        'The following is a conversation between a customer and a support agent...'
    ],
    'intent-label.output': [
        'The following is a conversation between a customer and a support agent...',
        '\ngoal-label: find plan',
        'The following is a conversation between a customer and a support agent...',
        '\ngoal-label: cancel plan'
    ]
}"""
df = pd.read_csv('/Users/jonathan/Downloads/data.csv')
#df = pd.DataFrame(data)

"""def create_pairwise_comparison_df(df):
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
                    comp1 = row1[col].values[0]
                    comp2 = row2[col].values[0]
                    comparison_rows.append([id_, model1, model2, comp1, comp2, None])  # None for preference initially

    # Create the new DataFrame from the list of rows
    comparison_df = pd.DataFrame(comparison_rows, columns=['id', 'model1', 'model2', 'comp1', 'comp2', 'preference'])
    
    return comparison_df"""

def create_pairwise_comparison_df(df):
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

def truncate_to_words(text, num_words=10):
    if isinstance(text, float) and np.isnan(text):
        return ''
    words = text.split()
    return ' '.join(words[:num_words])

# Create the pairwise comparison DataFrame
comparison_df = create_pairwise_comparison_df(df)

# Iterate over each row and print the required information with truncated values
for index, row in comparison_df.iterrows():
    comp1_value_truncated = truncate_to_words(row['comp1.value'])
    comp2_value_truncated = truncate_to_words(row['comp2.value'])
    print(f"id: {row['id']}")
    print(f"model1: {row['model1']}, comp1: {row['comp1.name']} = {comp1_value_truncated}")
    print(f"model2: {row['model2']}, comp2: {row['comp2.name']} = {comp2_value_truncated}")
    print()  # Print a blank line for better readability