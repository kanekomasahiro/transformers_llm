# Simple LLMs with transformers

## Introduction

This project provides a straightforward implementation of Large Language Models (LLM) using the [Transformers](https://github.com/huggingface/transformers) library by Hugging Face.

## Installation

- Python Version: The code is compatible with Python version `3.9.5`.
- Dependencies: Install all the necessary dependencies using the following command:
```sehll
pip install -r requirements.txt
```

## Usage

To run the model, use the following command. Ensure to replace `path/to/your/data` and `path/to/your/output/dir` with your actual data and output directories.

```shell
export CUDA_VISIBLE_DEVICES=0
python src/llm.py --data_path path/to/your/data --save_path path/to/your/output/dir
```

## License

Refer to the LICENSE file in this repository for licensing information.