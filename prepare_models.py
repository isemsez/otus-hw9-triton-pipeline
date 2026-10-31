"""Prepare all model artifacts for the Triton model repository.

Produces:
  model_repository/text_embedder/1/model.onnx
      all-MiniLM-L6-v2 transformer exported to ONNX (feature extraction).
      Inputs: input_ids/attention_mask/token_type_ids INT64 [N, L],
      output: last_hidden_state FP32 [N, L, 384]. Mean pooling is done
      by the BLS pipeline at request time.

  model_repository/text_classifier/1/model.onnx
      LogisticRegression over the 384-dim embedding, question vs statement,
      exported with skl2onnx. Inputs: embedding FP32 [N, 384],
      outputs: label INT64 [N], probabilities FP32 [N, 2].

  model_repository/pipeline/1/artifacts/tokenizer/
      MiniLM tokenizer used by the BLS script.

  model_repository/text_generator/1/artifacts/qwen/
      Qwen2.5-0.5B-Instruct weights loaded by the Python backend.

Label convention: 0 = statement, 1 = question.
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "model_repository")
EMBEDDER_DIR = os.path.join(REPO, "text_embedder", "1")
CLASSIFIER_DIR = os.path.join(REPO, "text_classifier", "1")
TOKENIZER_DIR = os.path.join(REPO, "pipeline", "1", "artifacts", "tokenizer")
QWEN_DIR = os.path.join(REPO, "text_generator", "1", "artifacts", "qwen")

MAX_LEN = 128
EMB_DIM = 384

QUESTIONS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "When was the United Nations founded?",
    "Who wrote Pride and Prejudice?",
    "Why is the sky blue?",
    "Where is the Great Barrier Reef located?",
    "Which planet is the largest in the solar system?",
    "Can you explain quantum entanglement?",
    "Do penguins live in the Arctic?",
    "Is Python a compiled language?",
    "What causes earthquakes?",
    "How many bones are in the human body?",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "When did the Berlin Wall fall?",
    "How far is the Moon from Earth?",
    "What languages are spoken in Switzerland?",
    "Could you tell me the time?",
    "Would you recommend this book?",
    "Has the meeting been postponed?",
    "Are the results available yet?",
    "What is machine learning?",
    "How do vaccines work?",
    "Why do we dream?",
    "Which countries border Brazil?",
    "Who invented the telephone?",
    "What does DNA stand for?",
    "How long is the Great Wall of China?",
    "Can machines think?",
    "Is there life on Mars?",
    "What are the symptoms of flu?",
    "How does the internet work?",
    "When will the train arrive?",
    "Where can I buy tickets?",
    "Who is the current president of France?",
    "What time does the store open?",
    "How much does it cost?",
    "Why did the stock market fall?",
    "Which team won the championship?",
    "Do you know the answer?",
    "May I ask you something?",
    "Should I take an umbrella today?",
    "Will it rain tomorrow?",
    "What is your favorite color?",
    "How old is the universe?",
]

STATEMENTS = [
    "The capital of France is Paris.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The United Nations was founded in 1945.",
    "Jane Austen wrote Pride and Prejudice.",
    "The sky appears blue because of Rayleigh scattering.",
    "The Great Barrier Reef is off the coast of Australia.",
    "Jupiter is the largest planet in the solar system.",
    "Penguins mostly live in the Southern Hemisphere.",
    "Python is an interpreted language.",
    "Earthquakes are caused by tectonic plate movements.",
    "The human body has 206 bones.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "The Berlin Wall fell in 1989.",
    "The Moon is about 384 thousand kilometers from Earth.",
    "Switzerland has four official languages.",
    "Machine learning is a subset of artificial intelligence.",
    "Vaccines train the immune system to fight pathogens.",
    "Scientists still debate the purpose of dreams.",
    "Brazil borders ten countries.",
    "Alexander Graham Bell patented the telephone.",
    "DNA stands for deoxyribonucleic acid.",
    "The Great Wall of China is over 21 thousand kilometers long.",
    "The meeting has been postponed to Friday.",
    "The results will be published next week.",
    "I need to buy groceries on my way home.",
    "This book is a fascinating read.",
    "She enjoys hiking in the mountains every summer.",
    "The company announced a new product line yesterday.",
    "Coffee is one of the most popular beverages in the world.",
    "My brother studies computer science at the university.",
    "The weather forecast predicts sunshine for the weekend.",
    "He finished the marathon in under four hours.",
    "The library closes at nine in the evening.",
    "Our team won the championship last season.",
    "The train to Boston departs from platform five.",
    "Tickets are available at the front desk.",
    "The stock market experienced significant volatility this month.",
    "I believe artificial intelligence will transform many industries.",
    "The museum exhibits ancient Roman artifacts.",
    "Breakfast is served until ten in the morning.",
    "Her presentation impressed the entire audience.",
    "The new bridge reduced travel time significantly.",
    "Autumn is my favorite season of the year.",
    "The documentary explored deep sea ecosystems.",
]


def export_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    print("== Exporting all-MiniLM-L6-v2 to ONNX ==")
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()

    dummy = tokenizer(
        ["a sample sentence"], padding="max_length", max_length=32, return_tensors="pt"
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            os.path.join(EMBEDDER_DIR, "model.onnx"),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=17,
            dynamo=False,
        )
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"saved: {EMBEDDER_DIR}/model.onnx")
    print(f"saved tokenizer: {TOKENIZER_DIR}")


def embed_texts(texts, onnx_path):
    """Embed texts with the exported ONNX model, mirroring the BLS logic:
    tokenizer -> ONNX -> mask-weighted mean pooling over last_hidden_state."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np"
    )
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc["token_type_ids"].astype(np.int64),
    }
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (names, _, _) = zip(*[(i.name, i.type, i.shape) for i in session.get_inputs()])
    print("ONNX inputs:", dict(zip(names, [i.type for i in session.get_inputs()])))
    hidden = session.run(None, feeds)[0]
    mask = enc["attention_mask"][:, :, None].astype(np.float32)
    emb = (hidden.astype(np.float32) * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    return emb


def export_classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    print("== Training question/statement classifier ==")
    texts = QUESTIONS + STATEMENTS
    y = np.array([1] * len(QUESTIONS) + [0] * len(STATEMENTS))

    onnx_path = os.path.join(EMBEDDER_DIR, "model.onnx")
    x = embed_texts(texts, onnx_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    acc = accuracy_score(y_test, clf.predict(x_test))
    print(f"held-out accuracy: {acc:.3f}")
    print(classification_report(y_test, clf.predict(x_test), target_names=["statement", "question"]))

    onx = to_onnx(
        clf,
        initial_types=[("embedding", FloatTensorType([None, EMB_DIM]))],
        options={id(clf): {"zipmap": False}},
        target_opset=17,  # Triton 24.12 ships ORT that supports opset <= 21
    )
    out_path = os.path.join(CLASSIFIER_DIR, "model.onnx")
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"saved: {out_path}")


def download_generator():
    from huggingface_hub import snapshot_download

    if os.path.exists(os.path.join(QWEN_DIR, "config.json")):
        print("== Qwen weights already present, skipping download ==")
        return
    print("== Downloading Qwen2.5-0.5B-Instruct (~1 GB) ==")
    snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", local_dir=QWEN_DIR)
    print(f"saved: {QWEN_DIR}")


if __name__ == "__main__":
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    export_embedder()
    export_classifier()
    download_generator()
    print("done.")
