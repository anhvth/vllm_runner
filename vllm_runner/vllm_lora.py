from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class LORA_VLLM:
    def __init__(self, model_name: str, lora_repo_id: str):
        # Download and store the LoRA adapter path
        self.lora_path = snapshot_download(repo_id=lora_repo_id)

        # Initialize the base model with LoRA support enabled
        self.llm = LLM(model=model_name, enable_lora=True)

    def generate(
        self,
        prompts,
        adapter_name: str,
        adapter_id: int,
        temperature=0.7,
        max_tokens=256,
        stop_tokens=None,
    ):
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens or ["[/assistant]"],
        )

        # Create a LoRA request for the adapter
        lora_request = LoRARequest(adapter_name, adapter_id, self.lora_path)

        # Generate output with the specified prompts
        outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_request)

        return outputs
