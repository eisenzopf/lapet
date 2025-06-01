class LiteLLMModel():
    def __init__(self, model_id):
        self.model_id = model_id

    def completion(self, messages, max_tokens, temperature, top_p):
        import litellm
        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        prompt, responses = response.choices[0].message.content, response.choices[0].message.content
        return prompt, responses

class LiteLLMModelHandler():
    """
    Model handler for LiteLLM models. Uses litellm.completion to generate outputs.
    Assumes API key is set in environment variables (e.g., OPENAI_API_KEY).
    """
    def load_model_and_tokenizer(self, device, model_id):
        self.model_id = model_id
        return None, LiteLLMModel(model_id)