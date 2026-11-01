"""Send one request through the BLS pipeline and print the full result."""

import argparse

import numpy as np
import tritonclient.http as httpclient


def run(text: str, url: str = "localhost:8000"):
    client = httpclient.InferenceServerClient(url=url)
    if not client.is_server_ready():
        raise SystemExit(f"Triton at {url} is not ready")

    inp = httpclient.InferInput("TEXT", [1], "BYTES")
    inp.set_data_from_numpy(np.array([text], dtype=np.object_))

    result = client.infer("pipeline", [inp])
    label = result.as_numpy("LABEL")[0].decode("utf-8")
    probs = result.as_numpy("PROBS")
    answer = result.as_numpy("ANSWER")[0].decode("utf-8")

    print(f"text      : {text}")
    print(f"label     : {label}")
    print(f"probs     : statement={probs[0]:.4f} question={probs[1]:.4f}")
    print(f"answer    : {answer or '<skipped: not a question>'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="What is the capital of France?")
    parser.add_argument("--url", default="localhost:8000")
    args = parser.parse_args()
    run(args.text, args.url)
