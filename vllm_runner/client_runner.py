from turtle import pd
from speedy_utils import *
import openai
from vllm_runner import scan_vllm_process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--input_df", type=str, default="input_df.csv")
parser.add_argument("--column", type=str, default="prompt")
parser.add_argument("--fold", nargs=2, type=int, default=[0, 1])
parser.add_argument("--cache_dir", type=str, default="cache")
args = parser.parse_args()

processes = scan_vllm_process()
# client = openai.OpenAI(base_url=f"http://localhost:{args.port}/v1")

if args.input_df.endswith(".csv"):
    df = pd.read_csv(args.input_df)
elif args.input_df.endswith(".parquet"):
    df = pd.read_parquet(args.input_df)


if "prompt" in args.column.lower():
    prompts = df[args.column].tolist()
    list_msgs = [{"role": "user", "content": prompt} for prompt in prompts]
elif "messages" in args.column or "msgs" in args.column:
    list_msgs = df[args.column].tolist()
    if hasattr(list_msgs[0], "tolist"):
        list_msgs = [msg.tolist() for msg in list_msgs]
else:
    raise ValueError(f"Column {args.column} not recognized")


