from collections import Counter
from typing import Dict, List, Literal

from dspy import LM
from loguru import logger
from speedy_utils import Clock
from transformers import AutoTokenizer
from vllm_runner.scan_vllm_process import scan_vllm_process

# from tools.server import NUM_SERVERS, SERVER_BASE_PORT

# Known models mapping
# KNOWN_MODELS = {
#     "QW72B": "Qwen2.5-72B-Instruct-AWQ",
#     "QW32B": "Qwen2.5-32B-Instruct-AWQ",
#     "QW7B": "Qwen2.5-7B-Instruct-AWQ",
#     "QW14B": "Qwen2.5-14B-Instruct-AWQ",
#     "QW1.5B": "Qwen2.5-1.5B-Instruct-AWQ",
#     "QW1.5B-Math": "Qwen/Qwen2.5-Math-1.5B-Instruct",
#     "QW7B": "Qwen2.5-7B-Instruct-AWQ",
#     "QW7B-Math": "Qwen/Qwen2.5-Math-7B-Instruct",
# }
clock = Clock()
KNOWN_MODELS = {}

class LLM(LM):
    def __init__(
        self,
        model: Literal["QW72B", "QW7B", "QW32B"] | str,
        max_tokens=2048,
        **kwargs,
    ):
        model_name = KNOWN_MODELS.get(model, model)
        self.scaned_ports = LLM.scan_model_ports()
        ports = self.scaned_ports.get(model_name, [-1])
        self.counter = Counter({port: 0 for port in ports})

        base_url = f"http://localhost:{ports[0]}/v1/"

        print(f"Using model 'openai/{model_name}', {self.counter}")
        kwargs["api_key"] = "abc"
        super().__init__(
            model=f"openai/{model_name}",
            base_url=base_url,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_url(self):
        port = self.counter.most_common()[-1][0]
        return f"http://localhost:{port}/v1/", port

    def count_tokens(self, messages) -> int:
        if not isinstance(messages, list) or not isinstance(messages[0], dict):
            messages = [{"role": "user", "content": str(messages)}]
        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B", trust_remote_code=True
            )
        return len(self.tokenizer.apply_chat_template(messages))

    def __call__(self, prompt=None, messages=None, **kwargs):
        if not hasattr(self, "counter"):
            return super().__call__(prompt=prompt, messages=messages, **kwargs)
        try:
            kwargs["base_url"], port = self.get_url()
            self.counter[port] += 1
            result = super().__call__(prompt, messages, **kwargs)
        except KeyboardInterrupt:
            raise
        finally:
            self.counter[port] -= 1
            if clock.time_since_last_checkpoint() > 5:
                logger.info(
                    f"Request on {port=} completed | {self.counter} | Num calls: {len(self.history)}"
                )
                clock.tick()
        return result

    @staticmethod
    def scan_model_ports() -> Dict[str, List[int]]:
        """
        Uses scan_vllm_process to find available models and returns a dictionary
        mapping model names to the ports where they are available.
        """
        model_to_servers: Dict[str, List[int]] = {}
        processes = scan_vllm_process()

        for process in processes:
            if process["model_name"] and process["port"]:
                port = int(process["port"])
                model_to_servers.setdefault(process["model_name"], []).append(port)

        logger.info(f"Scanned available models and their ports:\n{model_to_servers}")
        return model_to_servers

    # def ih(self, i=-1):
    #     from llm_utils import inspect_msgs

    #     res = self.history[i]["response"].choices[0].message.content
    #     msgs = self.history[i]["messages"] + [{"role": "assistant", "content": res}]
    #     return inspect_msgs(msgs)

    # def response(self, i):
    #     res = self.history[i]["response"].choices[0].message.content
    #     return res

    # def count_history_tokens(self, i):
    #     res = self.history[i]["response"].choices[0].message.content
    #     msgs = self.history[i]["messages"] + [{"role": "assistant", "content": res}]
    #     return self.count_tokens(msgs)
