import json
import os
import torch


def run(model, input_json_path, output_file="/tmp/last_run_vllm_embed.pt", **kwargs):
    output_file = os.path.abspath(output_file)
    from vllm_runner.overwrite_qwen_for_embedding import run_embedding

    with open(input_json_path) as f:
        data = json.load(f)
        assert input_json_path.endswith(
            ".json"
        ), "Input file must be a JSON file. Contain list of `texts`."
        if "texts" not in data:
            raise ValueError("Input JSON must contain a 'texts' key.")
        inputs = data["texts"]
    text2embed = run_embedding(model, inputs, **kwargs)
    torch.save(text2embed, output_file)
    print(f"{output_file=}")


if __name__ == "__main__":
    # Example
    import fire

    fire.Fire(run)
