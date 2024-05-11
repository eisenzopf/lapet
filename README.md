# LLM-Eval

Benchmark datasets like Bigbench and public benchmark leaderboards like Huggingface LLM Leaderboard are great for getting a sense of the kinds of tests foundation model builders use and which models perform well. However, this is not particularly helpful for users, who need to compare the results of models for specific LLM generative tasks like seeing how well a model is at summarizing a customer service call or putting together an action plan to resolve a customer issue or analyzing a spreadsheet for inconsistencies. These real world tasks require an evaluation method that is easy enough for users to evaluate which model will perform best for them or to even evaluate the results of a model finetuning. 

The purpose of this library is to make it easier to evaluate the quality of LLM outputs from multiple models across a set of user selectable tasks.

The metrics will measure for speed, model size (memory), and quality (accuracy). The system can utilize one or more LLMs as judges in addition to human judges. The results can be collected and used as training data. Users can select a public dataset and task or use their own.

User inputs will be:
- list of models to evaluate
- list of prompt templates
- a dataset that will be combined with the prompt templates
- selection one one or more LLM judges
- option to include a human judge

Outputs will be in a csv format and will include graph(s) showing: 
- speed (tokens per second)
- memory usage (for models run locally)
- quality (as judged by LLM and human judges)
- Agreement cappa for judge results (if there is more than 1 judge)
