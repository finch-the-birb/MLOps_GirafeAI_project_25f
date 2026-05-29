"""Query deployed Triton multimodal model with synthetic inputs."""

from __future__ import annotations

import sys

import numpy as np
import tritonclient.http as httpclient


def main() -> int:
    triton_url = "localhost:8000"
    model_name = "multimodal_model_early"
    model_version = "1"

    time_series = np.random.randn(1, 30, 6).astype(np.float32)
    text_per_step = np.random.randn(1, 30, 32).astype(np.float32)

    client = httpclient.InferenceServerClient(url=triton_url, verbose=False)
    if not client.is_server_live():
        print("Triton server is not reachable.", file=sys.stderr)
        return 1

    inputs = [
        httpclient.InferInput("time_series", time_series.shape, "FP32"),
        httpclient.InferInput("text_per_step", text_per_step.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(time_series)
    inputs[1].set_data_from_numpy(text_per_step)
    outputs = [httpclient.InferRequestedOutput("probability")]

    response = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=inputs,
        outputs=outputs,
    )
    probability = response.as_numpy("probability")
    print(f"probability={probability.reshape(-1)[0]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
