import time

from .handler import ModelHandler
class LlamaModelHandler(ModelHandler):
      """Handler for Llama models specialized for causal language modeling."""
      """ for llama 3
      terminators = [
          self.model.tokenizer.eos_token_id,
          self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.model(prompt, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
      
      """
      def batch_outputs(self, strings):
        import torch
        inputs = self.tokenizer(strings, return_tensors="pt", padding=True, truncation=True, max_length=1200)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model.generate(**inputs, max_new_tokens=200)
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

      def batch_process(self, batch_size, text):
            import torch
            strings = []
            responses = []
            prompts = []
            counter = 0
            for ca_prompt in self.ca_prompts:
                  if counter == 0:
                        messages = [
                              {"role": "system", "content": self.system_prompt},
                              {"role": "user", "content": text + "\n\n" + ca_prompt['prompt']},
                        ]
                        prompt = self.tokenizer.apply_chat_template(
                              messages,
                              tokenize=False,
                              add_generation_prompt=True
                        )
                        strings.append(prompt)
                        prompts.append(prompt)
                        counter += 1
                  elif counter == batch_size:
                        responses.extend(self.batch_outputs(strings))
                        counter = 0
                        strings = []
                  else:
                        messages = [
                              {"role": "user", "content": ca_prompt['prompt']},
                        ]
                        prompt = self.tokenizer.apply_chat_template(
                              messages,
                              tokenize=False,
                              add_generation_prompt=True
                        )
                        strings.append(prompt)
                        prompts.append(prompt)
                        counter += 1

            if len(strings) > 0:
                  responses.extend(self.batch_outputs(strings))

            return prompts, responses
      
      def generate_output(self, texts):
        total_tokens = 0
        total_time = 0
        results = {}
        for id in texts:
            start_time = time.time()
            prompts, responses = self.batch_process(3, texts[id])
            results[id] = responses
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            for prompt, response in zip(prompts, responses):
                tokens = self.tokenizer.tokenize(response)
                total_tokens += len(tokens)
                response = self.post_process_output(response[len(prompt)-1:])
                results[id] = response
                print(response)
                print("-------------------------------------------------------")

        avg_tokens_per_second = total_tokens / total_time
        #print(f"Average tokens per second for {self.model_id}: {total_tokens / total_time}")
        return avg_tokens_per_second, results