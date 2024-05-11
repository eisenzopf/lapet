import pandas as pd
import numpy as np
import huggingface_hub
from llm_eval import LlamaModelHandler
if __name__ == "__main__":
    """Base class for handling different models with common interface for generating outputs."""
    system_prompt = '''The following is a conversation between a customer service agent who works at Union Mobile and a customer who needs support.
    The customer is talking when the line starts with the word customer. The agent is talking when the line starts with the word agent.
    First, read the conversation carefully. Then read the question and think about how to answer it. Then provide your answer. Be concise.'''

    prompts = [
        {"name": "intent-sentence", "prompt": '''In a sentence, describe the customer's goal of the conversation. Format the output using the following format: { "goal": "<goal>" }'''},
        {"name": "intent-label", "prompt": '''Create a two word label for the customer's goal where the first word is a verb and the second word is a noun. Use the following output format: { 'goal-label': '[VERB NOUN]' }'''},
        {"name": "customer-sentiment", "prompt": '''What is the customer's sentiment? Pick one: Positive, Neutral, Negative. Use the following output format: { "customer-sentiment": "<sentiment>" }'''},
        {"name": "agent-empathy", "prompt": '''What was the agent's level of empathy? Pick one: Poor, Average, High. Use the following output format: { "agent-empathy": "<empathy score>" }'''},
        {"name": "outcome", "prompt": '''Did the customer achieve their goal? Yes or no. Explain. Use the following output format: { "Goal Achieved": "<yes or no>", "Explanation": "<explanation>" }'''},
        {"name": "summary", "prompt": '''Write a summary of the conversation including the goal that the customer wanted to accomplish, the actions that the agent took, and the outcome of the conversation in no more than 5 sentences. Use the following output format: { "summary": "<summary>" }'''}
    ]

    models = {
        #'microsoft/Phi-3-mini-4k-instruct': PhiModelHandler,
        'meta-llama/Llama-2-7b-chat-hf': LlamaModelHandler,
        #'meta-llama/Meta-Llama-3-8B-Instruct': Llama3ModelHandler,
        #'MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ': GPTQModelHandler,
        #'FriendliAI/Meta-Llama-3-8B-Instruct-fp8': FP8ModelHandler,
        #'solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ': AWQModelHandler,
    }

    config = {
        "batch_size": 3,
        "max_length": 1200,
        "max_new_tokens": 100,
        "temperature":0.6,
        "top_p": 0.9,
        "dataset": "https://huggingface.co/datasets/talkmap/telecom-conversation-corpus/resolve/main/telecom_200k.csv",
        "samples": 20,
        "models": models,
        "system_prompt": system_prompt,
        "prompts": prompts,
    }

    huggingface_hub.login()
    # replace
    handler = LlamaModelHandler("meta-llama/Llama-2-7b-chat-hf", config)
    responses = handler.process_dataset()
    responses.to_csv('updated_data_output.csv', index=False)
    #for response in responses:
    #    print(response)
    handler.unload_model()