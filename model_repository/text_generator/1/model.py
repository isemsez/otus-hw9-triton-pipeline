"""text_generator: Qwen2.5-0.5B-Instruct served via the Triton Python backend.

Receives a raw user string, applies the chat template, generates a short
greedy completion on CPU and returns the decoded answer.
"""

import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "qwen")
MAX_NEW_TOKENS = 48


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS)
        self.model = AutoModelForCausalLM.from_pretrained(ARTIFACTS, torch_dtype=torch.float32)
        self.model.eval()
        # leave one core for the embedder/classifier/Triton itself
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    def execute(self, requests):
        responses = []
        for request in requests:
            text = request.inputs()[0].as_numpy()[0].decode("utf-8")
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            answer = self.tokenizer.decode(
                tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            out = pb_utils.Tensor("GENERATED", np.array([answer], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
