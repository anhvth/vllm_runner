#!/usr/bin/env python3

import json
import os
from loguru import logger
import pandas as pd
from speedy_utils import *
import openai
from vllm_runner import scan_vllm_process
import argparse
from vllm_runner.vllm_dspy import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--input_df", type=str, default="input_df.csv")
    parser.add_argument("-c", "--column", type=str, default="prompt")
    parser.add_argument("-f", "--fold", nargs=2, type=int, default=[0, 1])
    parser.add_argument("-cd", "--cache_dir", type=str, default="cache")
    parser.add_argument("-m", "--model", type=str, default="QW32B")
    parser.add_argument("-mt", "--max_tokens", type=int, default=2048)
    parser.add_argument("-ip", "--is_prompt", action="store_true")
    return parser.parse_args()


def construc_cache_path(cache_dir, i):
    return os.path.join(cache_dir, f"{i}.json")


def run_one_msgs(lm, msgs, max_tokens):
    out = lm(messages=msgs, max_tokens=max_tokens)[0]
    msgs = msgs + [{"role": "assistant", "content": out}]
    return msgs


def run_one_row(row, args, lm, cache_dir, is_prompt):
    row_id = row["id"]
    cache_path = construc_cache_path(cache_dir, row_id)
    if os.path.exists(cache_path):
        logger.info(f"Cache exists for index {row_id}, skipping... {cache_path}")
        return

    if is_prompt:
        msgs = [{"role": "user", "content": row[args.column]}]
    else:
        msgs = row[args.column]
        if hasattr(msgs, "tolist"):
            msgs = msgs.tolist()

    msgs = run_one_msgs(lm, msgs, args.max_tokens)
    dump_json_or_pickle(msgs, cache_path)
    logger.success(f"Finished processing index {row_id}, result cached. {cache_path}")


def main():
    args = parse_args()

    if args.input_df.endswith(".csv"):
        df = pd.read_csv(args.input_df)
    elif args.input_df.endswith(".parquet"):
        df = pd.read_parquet(args.input_df)

    df["id"] = df.index
    df = df[args.fold[0] :: args.fold[1]]

    if not args.column in df.columns:
        raise ValueError(
            f"Column {args.column} not found in the dataframe\n{df.columns}"
        )

    is_prompt = "prompt" in args.column.lower() or args.is_prompt

    lm = LLM(model=args.model)
    file_name = os.path.basename(args.input_df).split(".")[0]
    cache_dir = os.path.join(args.cache_dir, args.model, file_name)
    os.makedirs(cache_dir, exist_ok=True)

    rows = df.to_dict(orient="records")
    output_rows = multi_thread(
        lambda row: run_one_row(row, args, lm, cache_dir, is_prompt), rows, 4
    )
    df = pd.DataFrame(output_rows)
    output_file = (
        args.input_df
        + f".model_{args.model}_{args.column}_fold_{args.fold[0]}_{args.fold[1]}.parquet"
    )
    df.to_parquet(output_file)


if __name__ == "__main__":
    main()
