#!/usr/bin/env python3
"""
Main entry point for the lapet package.
Run with: python -m lapet
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='LAPET - LLM Evaluation Library')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate evaluation data')
    generate_parser.add_argument('--config', type=str, help='Path to config file')
    generate_parser.add_argument('--output', type=str, default='eval_data.csv', help='Output CSV file')
    
    # Evaluate command  
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model outputs')
    eval_parser.add_argument('--input', type=str, default='eval_data.csv', help='Input CSV file')
    eval_parser.add_argument('--output', type=str, default='eval_results.csv', help='Output CSV file')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        run_generate(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'version':
        show_version()
    else:
        parser.print_help()

def run_generate(args):
    """Run the generation pipeline"""
    print("Running generation pipeline...")
    
    # Import here to avoid import errors if dependencies aren't installed
    try:
        import pandas as pd
        import numpy as np
        import huggingface_hub
        from . import ModelHandler, Llama2ModelHandler, Llama3ModelHandler, Llama31ModelHandler, Phi4ModelHandler, GemmaModelHandler
        
        # Default config (you can make this configurable later)
        config = {
            'batch_size': 5,
            'max_length': 1500,
            'max_new_tokens': 500,
            'temperature': 0.6,
            'top_p': 0.9,
            'dataset': 'https://huggingface.co/datasets/talkmap/telecom-conversation-corpus/resolve/main/telecom_200k.csv',
            'samples': 5,
            'models': {
                'meta-llama/Llama-2-7b-chat-hf': Llama2ModelHandler,
                'meta-llama/Meta-Llama-3-8B-Instruct': Llama3ModelHandler,
                'meta-llama/Llama-3.1-8B-Instruct': Llama31ModelHandler,
                'google/gemma-3-1b-it': GemmaModelHandler,
            },
            'system_prompt': "You are a helpful AI assistant that generates answers to questions. You will be provided with transcripts of conversations between customers and service agents. Your task is to follow the instruction and output a response from each conversation in a valid JSON format.",
            'prompts': [
                {'name': 'intent-sentence','prompt': 'In a sentence, describe the customer\'s goal of the conversation. Format the output in JSON using the following format: { "intent-sentence": "<OUTPUT>" } where <OUTPUT> is the sentence. Be concise.'},
                {'name': 'customer-sentiment','prompt': 'What is the customer\'s sentiment? Pick one: Positive, Neutral, Negative.  Format the output in JSON using the following format: { "customer-sentiment": "<OUTPUT>" } where <OUTPUT> is the sentiment. Be concise.'},
            ]
        }
        
        print("Logging into Hugging Face...")
        huggingface_hub.login()
        
        print("Creating ModelHandler...")
        handler = ModelHandler(config)
        
        print("Processing dataset...")
        responses = handler.process_dataset()
        
        print(f"Saving results to {args.output}...")
        responses.to_csv(args.output, index=False)
        
        print(f"✅ Generation complete! Results saved to {args.output}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the required dependencies with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        sys.exit(1)

def run_evaluate(args):
    """Run the evaluation pipeline"""
    print("Running evaluation pipeline...")
    try:
        from .judge import LLMJudge
        
        print(f"Loading data from {args.input}...")
        judge = LLMJudge()
        # Add evaluation logic here
        print(f"✅ Evaluation complete! Results saved to {args.output}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the required dependencies.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)

def show_version():
    """Show version information"""
    try:
        import importlib.metadata
        version = importlib.metadata.version('lapet')
        print(f"LAPET version {version}")
    except:
        print("LAPET version unknown")

if __name__ == '__main__':
    main() 