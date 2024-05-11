import huggingface_hub
from llm_eval import ModelHandler

dataset = "https://huggingface.co/datasets/talkmap/telecom-conversation-corpus/resolve/main/telecom_200k.csv",
results = ModelHandler.load_dataset(dataset)