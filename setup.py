from setuptools import setup, find_packages

setup(
    name="vllm_runner",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gradio",
        "loguru==0.7.2",
        "numpy==1.26.4",
        "requests==2.32.3",
        "xxhash==3.5.0",
        "fastcore==1.7.5",
        "debugpy==1.8.6",
        "ipywidgets==8.1.5",
        "jupyterlab==4.2.5",
        "ipdb==0.13.13",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "pandas==2.2.2",
        "tabulate==0.9.0",
        "pydantic==2.9.1",
        "speedy-utils",
        "python-dotenv",
        "openai",
        "dspy-ai",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "vllm-client=vllm_runner.scripts.client_runner:main",
            "vllm-server=vllm_runner.scripts.vllm_server:main",
        ],
    },
)
