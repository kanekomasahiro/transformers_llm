from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import fire
import pickle
import torch
import pandas as pd


class DataManager():

    def __init__(self, data_path: str, save_path: str, batch_size: int):
        self.data_path = data_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.set_data_loader()

    def __iter__(self):
        return iter(self.data_loader)


    def __len__(self):
        return len(self.data_loader)

    def load_data(self):
        if self.data_path.endswith((".csv", ".json")):
            data = pd.read_csv(self.data_path) if self.data_path.endswith(".csv") else json.load(open(self.data_path, "r"))
        elif self.data_path.endswith((".txt", ".jsonl")):
            with open(self.data_path, "r") as f:
                data = f.readlines() if self.data_path.endswith(".txt") else [json.loads(line) for line in f.readlines()]
        else:
            raise ValueError("Data path must be csv, json, txt, or jsonl")
        return data

    def save_data(self, data):
        if self.save_path.endswith((".csv", ".json")):
            if self.save_path.endswith(".csv"):
                data.to_csv(self.save_path)
            else:
                with open(self.save_path, "w") as f:
                    json.dump(data, f)
        elif self.save_path.endswith((".txt", ".jsonl")):
            with open(self.save_path, "w") as f:
                if self.save_path.endswith(".txt"):
                    f.write(data)
                else:
                    for line in data:
                        f.write(json.dumps(line))
                        f.write("\n")
        elif self.save_path.endswith(".pkl"):
            with open(self.save_path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Save path must be csv, json, txt, jsonl, or pkl")

    def set_data_loader(self):
        data = self.load_data()
        self.data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)


class Pipeline:

    def __init__(self, model_path, input_length, output_length, num_beams, dtype):
        self.device = self.initialize_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, max_length=input_length, padding="max_length", truncation=True, padding_side="left")
        self.tokenizer.pad_token = "<PAD>"
        config = self.get_model_config(model_path, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(**config)
        self.model.to(self.device)
        self.output_length = output_length
        self.num_beams = num_beams

    def calculate_max_memory(self):
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f'{free_in_GB - 2}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        return max_memory

    def get_model_config(self, model_path, dtype):
        if dtype == "float32":
            return {"pretrained_model_name_or_path": model_path, "output_hidden_states": True, "torch_dtype": torch.float32}
        elif dtype == "float16":
            return {"pretrained_model_name_or_path": model_path, "output_hidden_states": True, "torch_dtype": torch.float16}
        elif dtype == "bfloat16":
            return {"pretrained_model_name_or_path": model_path, "output_hidden_states": True, "torch_dtype": torch.bfloat16}
        elif dtype == "8bit":
            max_memory = self.calculate_max_memory()
            return {"pretrained_model_name_or_path": model_path, "output_hidden_states": True, "load_in_8bit": True, "device_map": "auto", "max_memory": max_memory}
        elif dtype == "4bit":
            max_memory = self.calculate_max_memory()
            return {"pretrained_model_name_or_path": model_path, "output_hidden_states": True, "load_in_4bit": True, "device_map": "auto", "max_memory": max_memory, "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": torch.bfloat16}
        else:
            raise ValueError("dtype must be float32, float16, bfloat16, 8bit, or 4bit")

    def initialize_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def tokenize(self, batch):
        return self.tokenizer(batch, padding=True, return_tensors="pt").to(self.device)

    def detokenize(self, generate_id):
        return self.tokenizer.batch_decode(generate_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def likelihood(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Calculate the likelihood of the batch.

        Args:
            batch (torch.Tensor): A tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of shape (batch_size,).
        """
        encoded_batch = self.tokenize(batch)
        attention_mask = encoded_batch.attention_mask
        outputs = self.model(**encoded_batch)
        logits = outputs.logits
        all_scores = logits.gather(-1, encoded_batch.input_ids.unsqueeze(-1)).squeeze()
        unmasked_scores = all_scores * attention_mask
        avg_likelihood = unmasked_scores.sum(1) / attention_mask.sum(1)
        return avg_likelihood

    def embedding(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Calculate the embedding of the batch.

        Args:
            batch (torch.Tensor): A tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size).
        """
        encoded_batch = self.tokenize(batch)
        attention_mask = encoded_batch.attention_mask.unsqueeze(-1)
        outputs = self.model(**encoded_batch)
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[-1]  # last layer
        unmasked_embeddings = embeddings * attention_mask
        avg_embeddings = unmasked_embeddings.sum(1) / attention_mask.sum(1)
        return avg_embeddings

    def generate(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate text from the batch.

        Args:
            batch (torch.Tensor): A tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length).
        """
        encoded_batch = self.tokenize(batch)
        generate_id = self.model.generate(encoded_batch.input_ids, max_length=self.output_length, num_beams=self.num_beams)
        generate_text = self.detokenize(generate_id)
        return generate_text


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def main(data_path: str,
         save_path: str,
         model_path: str = "meta-llama/Llama-2-7b-chat-hf",
         mode: str = "generate",
         dtype: str = "float16",
         batch_size: int = 2,
         input_length: int = 128,
         output_length: int = 128,
         num_beams: int = 5):
    
    datamanager = DataManager(data_path, save_path, batch_size)
    pipeline = Pipeline(model_path, input_length, output_length, num_beams, dtype)
    outputs = []

    for batch in tqdm(datamanager):
        if mode == "likelihood":
            output = pipeline.likelihood(batch)
        elif mode == "embedding":
            output = pipeline.embedding(batch)
        elif mode == "generate":
            output = pipeline.generate(batch)
        outputs.append(output)

    if mode in ["likelihood", "embedding"]:
        outputs = to_numpy(torch.cat(outputs, dim=0))
    elif mode == "generate":
        outputs = [item for sublist in outputs for item in sublist]

    datamanager.save_data(outputs)


if __name__ == "__main__":
    fire.Fire(main)
