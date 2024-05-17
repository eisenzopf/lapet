import pandas as pd
import os
from llm_eval import LLMJudge

config = {
    "judge": {
        "name": "openai",
        "organization": os.environ["OPENAI_ORG"],
        "project": os.environ["OPENAI_PROJECT"],
        "api_key": os.environ["OPENAI_KEY"]
    }
}

df = pd.read_csv('data.csv')
llm_judge = LLMJudge(config, df)
llm_judge.evaluate()
llm_judge.dataset.to_csv("judging_results.csv")