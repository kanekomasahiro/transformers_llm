#!/bin/sh

export CUDA_VISIBLE_DEVICES=2

python src/llm.py --data_path data/news-commentary-v18.en.txt --save_path outputs/sample.txt

