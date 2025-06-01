import pandas as pd
import os
from lapet import LLMJudge

config = {
    "judge": {
        "name": "openai",
        "organization": os.getenv("OPENAI_ORG"),
        "project": os.getenv("OPENAI_PROJECT"),
        "api_key": os.getenv("OPENAI_API_KEY")
    }
}

df = pd.read_csv('eval_data.csv')
llm_judge = LLMJudge(config, df)
llm_judge.evaluate()
llm_judge.dataset.to_csv("eval_results.csv")