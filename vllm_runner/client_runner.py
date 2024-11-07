import json
import os
from loguru import logger
import pandas as pd
from speedy_utils import *
import openai
from vllm_runner import scan_vllm_process
import argparse

from vllm_runner.vllm_dspy import LLM

parser = argparse.ArgumentParser()
parser.add_argument("-df", "--input_df", type=str, default="input_df.csv")
parser.add_argument("-c", "--column", type=str, default="prompt")
parser.add_argument("-f", "--fold", nargs=2, type=int, default=[0, 1])
parser.add_argument("-cd", "--cache_dir", type=str, default="cache")
parser.add_argument("-m", "--model", type=str, default="QW32B")
parser.add_argument("-mt", "--max_tokens", type=int, default=2048)
parser.add_argument("-ip", "--is_prompt", action="store_true")
args = parser.parse_args()


if args.input_df.endswith(".csv"):
    df = pd.read_csv(args.input_df)
elif args.input_df.endswith(".parquet"):
    df = pd.read_parquet(args.input_df)
df["id"] = df.index
df = df[args.fold[0] :: args.fold[1]]
ids = df.index.tolist()


if not args.column in df.columns:
    raise ValueError(f"Column {args.column} not found in the dataframe\n{df.columns}")

if "prompt" in args.column.lower() or args.is_prompt:
    IS_PROMPT = True
else:
    IS_PROMPT = False


lm = LLM(model=args.model)
file_name = os.path.basename(args.input_df).split(".")[0]
cache_dir = os.path.join(args.cache_dir, args.model, file_name)


def construc_cache_path(i):
    return os.path.join(cache_dir, f"{i}.json")


def run_one_msgs(msgs):
    out = lm(messages=msgs, max_tokens=args.max_tokens)[0]
    msgs = msgs + [{"role": "assistant", "content": out}]
    return msgs


def run_one_row(row):
    row_id = row["id"]
    cache_path = construc_cache_path(row_id)
    if os.path.exists(cache_path):
        
        logger.info(f"Cache exists for index {row_id}, skipping... {cache_path}")
        return
    if IS_PROMPT:
        msgs = [{"role": "user", "content": row[args.column]}]
    else:
        msgs = row[args.column]
        if hasattr(msgs, "tolist"):
            msgs = msgs.tolist()
    msgs = run_one_msgs(msgs)
    dump_json_or_pickle(msgs, cache_path)
    logger.success(f"Finished processing index {row_id}, result cached. {cache_path}")


rows = df.to_dict(orient="records")
output_rows = multi_thread(run_one_row, rows, 4)
df = pd.DataFrame(output_rows)
df_output_path = args.input_df.replace(".csv", f"_output_{args.model}_{args.fold[0]}_{args.fold[1]}.csv")
