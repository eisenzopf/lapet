import pandas as pd
import itertools
import numpy as np
import re

df = pd.read_csv('/Users/jonathan/Downloads/data.csv')

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

def extract_output_by_name(name, value):
    print(f"name: {name}\nvalue: {value}")
    if isinstance(value, float) and np.isnan(value):
        return ''
    
    # Search for the section following "### Output ###"
    match1 = re.search(r'### Output ###(.*?)', value, re.DOTALL)
    if match1:
        new_value = match1.group(1).strip()  # Extract and strip whitespace from the result
        # Search within the extracted section for a specific 'name' and its value
        match2 = re.search(rf'{name}:\s*(.*)', new_value, re.DOTALL)
        if match2:
            return match2.group(1).strip()  # Return the name's value, strip to clean it
        else:
            return new_value  # Return the new_value if no specific name is found
    return ''  # Return empty if no output section is found



# Create the pairwise comparison DataFrame
comparison_df = create_pairwise_comparison_df(df)

# Iterate over each row and print the required information with truncated values
for index, row in comparison_df.iterrows():
    comp1_value_truncated = extract_output_by_name(row['comp1.name'], row['comp1.value'])
    comp2_value_truncated = extract_output_by_name(row['comp1.name'], row['comp2.value'])
    print(f"id: {row['id']}")
    print(f"model1: {row['model1']}, comp1 - {row['comp1.name']}: {comp1_value_truncated}")
    print(f"model2: {row['model2']}, comp2 - {row['comp2.name']}: {comp2_value_truncated}")
    print()  # Print a blank line for better readability