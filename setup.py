from setuptools import setup, find_packages

setup(
    name="vllm_runner",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gradio",
        "loguru",
    ],
)
