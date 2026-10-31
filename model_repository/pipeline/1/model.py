"""pipeline: BLS orchestrator (Business Logic Scripting) on the Python backend.

Per request:
  1. tokenize the input text with the MiniLM tokenizer;
  2. call text_embedder (ONNX) and mean-pool last_hidden_state;
  3. call text_classifier (ONNX) on the embedding;
  4. if the text is classified as a question, call text_generator (LLM);
  5. assemble LABEL / PROBS / ANSWER into one response.
"""

import os

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

TOKENIZER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "artifacts", "tokenizer"
)
MAX_LEN = 128
LABELS = {0: "statement", 1: "question"}


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def _embed(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np"
        )
        inputs = [
            pb_utils.Tensor("input_ids", enc["input_ids"].astype(np.int64)),
            pb_utils.Tensor("attention_mask", enc["attention_mask"].astype(np.int64)),
            pb_utils.Tensor("token_type_ids", enc["token_type_ids"].astype(np.int64)),
        ]
        request = pb_utils.InferenceRequest(
            model_name="text_embedder",
            model_version=-1,
            requested_output_names=["last_hidden_state"],
            inputs=inputs,
        )
        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonError(f"text_embedder failed: {response.error().message}")
        hidden = pb_utils.get_output_tensor_by_name(response, "last_hidden_state").as_numpy()
        mask = enc["attention_mask"][:, :, None].astype(np.float32)
        emb = (hidden.astype(np.float32) * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return emb.astype(np.float32)

    def _classify(self, embedding):
        request = pb_utils.InferenceRequest(
            model_name="text_classifier",
            model_version=-1,
            requested_output_names=["probabilities"],
            inputs=[pb_utils.Tensor("embedding", embedding)],
        )
        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonError(f"text_classifier failed: {response.error().message}")
        return pb_utils.get_output_tensor_by_name(response, "probabilities").as_numpy()

    def _generate(self, text):
        request = pb_utils.InferenceRequest(
            model_name="text_generator",
            model_version=-1,
            requested_output_names=["GENERATED"],
            inputs=[pb_utils.Tensor("TEXT", np.array([text], dtype=np.object_))],
        )
        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonError(f"text_generator failed: {response.error().message}")
        return pb_utils.get_output_tensor_by_name(response, "GENERATED").as_numpy()[0].decode(
            "utf-8"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            text = request.inputs()[0].as_numpy()[0].decode("utf-8")

            embedding = self._embed([text])
            probs = self._classify(embedding)[0]
            label_id = int(np.argmax(probs))
            answer = ""
            if label_id == 1:  # question -> ask the LLM
                answer = self._generate(text)

            outputs = [
                pb_utils.Tensor(
                    "LABEL", np.array([LABELS[label_id]], dtype=np.object_)
                ),
                pb_utils.Tensor("PROBS", probs.astype(np.float32)),
                pb_utils.Tensor("ANSWER", np.array([answer], dtype=np.object_)),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
