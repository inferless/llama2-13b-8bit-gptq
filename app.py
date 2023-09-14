import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import argparse

model_name_or_path = "TheBloke/Llama-2-13B-GPTQ"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="gptq-8bit-64g-actorder_True")


    def infer(self, inputs):
        prompt = inputs["prompt"]
        prompt_template=f'''USER: {prompt}
        ASSISTANT:'''

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        generated_text = pipe(prompt_template)[0]['generated_text']
        return {"generated_text": generated_text}

    def finalize(self):
        self.tokenizer = None
        self.model = None
