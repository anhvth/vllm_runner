# Overwrite the forward function of Qwen2ForCausalLM to save hidden states to cache

import json
import os
import shutil
from glob import glob
from typing import Any, List, Optional

import torch
import vllm
from fastcore.all import patch
from speedy_utils import identify, load_json_or_pickle
from tqdm import tqdm
from vllm.model_executor.models.qwen2 import (
    AttentionMetadata,
    IntermediateTensors,
    Qwen2ForCausalLM,
)

CACHE_DIR = os.path.expanduser("~/.cache/vllm_runner")

current_cache_dir = None


def clean_cache(cache_dir):
    print("Clean cache:", cache_dir)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)


@patch
def forward(
    self: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> torch.Tensor:
    global current_cache_dir
    hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)

    idx = identify(input_ids)
    torch.save(
        {
            "input_ids": input_ids.cpu(),
            "hidden_states": hidden_states.half().cpu(),
        },
        f"{current_cache_dir}/{idx}.pt",
    )
    return hidden_states


def load_and_split_data(tokenizer):
    global current_cache_dir

    def load_cache():
        files = glob(f"{current_cache_dir}/*.pt")
        items = []
        for file in files:
            data = torch.load(file)
            items.append(data)
        return items

    data = load_cache()

    # Concatenate all data
    input_ids = torch.cat([item["input_ids"] for item in data], dim=0)
    hidden_states = torch.cat([item["hidden_states"] for item in data], dim=0)
    positions = torch.cat([item["positions"] for item in data], dim=0)

    # Split by position
    def split_by_position(input_ids, hidden_states, positions):
        splits = []
        start = 0
        for i in range(1, len(positions)):
            if positions[i] == 0:
                splits.append((input_ids[start:i], hidden_states[start:i]))
                start = i
        # Add the last split
        splits.append((input_ids[start:], hidden_states[start:]))
        return splits

    splits = split_by_position(input_ids, hidden_states, positions)

    # Decode text and return splits
    list_splited_input_ids = [item[0] for item in splits]
    list_splited_hidden_states = [item[1] for item in splits]
    list_splited_text = [tokenizer.decode(item[0]) for item in splits]

    return list_splited_input_ids, list_splited_text, list_splited_hidden_states


def run_embedding(
    model_path: str,
    inputs: list[str],
    clean_after_run: bool = True,
    tp=2,
):
    assert isinstance(inputs, list), "Input must be a list of texts."
    assert isinstance(inputs[0], str), "Input must be a list of texts."
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in devices])
    print("First text:", inputs[0])
    global current_cache_dir  # Add this line
    input_id = identify(inputs)
    current_cache_dir = os.path.join(CACHE_DIR, input_id)  # Update this line
    clean_cache(current_cache_dir)  # Update this line
    model = vllm.LLM(
        model_path,
        tensor_parallel_size=tp,
        max_model_len=1024,
        dtype=torch.float16,
        enforce_eager=True,
        quantization="awq" if "awq" in model_path.lower() else None,
    )
    tokenizer = model.get_tokenizer()

    print("Num input:", len(inputs))

    # Generate model outputs
    model.generate(  # return nothing but save cache
        inputs,
        sampling_params=vllm.SamplingParams(
            n=1,
            top_p=0.8,
            temperature=0.0,
            seed=777,
            skip_special_tokens=False,
            max_tokens=1,
        ),
    )

    # Load and split data
    _, list_splited_text, list_splited_hidden_states = load_and_split_data(tokenizer)

    # Check and map text to embedding
    text2embed = {}
    for text, hidden_states in zip(list_splited_text, list_splited_hidden_states):
        if text in inputs:
            text2embed[text] = hidden_states[-1]
    if clean_after_run:
        clean_cache(current_cache_dir)  # Update this line
    return text2embed


if __name__ == "__main__":
    # Example
    import fire

    fire.Fire(run_embedding)
