# %%
import copy
import random
import re
import time
from typing import Any, List, Literal, Set, Tuple, TypedDict, Union
import torch.nn.init as init
import torch
import wandb
import transformers
import torch
from torch.utils.data import Dataset as TorchDataset
from functools import partial

# from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from jaxtyping import Int
from torch import FloatTensor, Tensor
from torch.optim import Adam
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from pathlib import Path

Path("./data/").mkdir(parents=True, exist_ok=True)
Path("./cache_dir/").mkdir(parents=True, exist_ok=True)


def check(error_type, condition, message):
    if condition:
        raise error_type(message)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models():
    tinystories_model = transformers.AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-1M", revision="8cd14d5", cache_dir="./data/"
    )

    random_init_model = transformers.AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-1M", revision="8cd14d5", cache_dir="./data/"
    )

    def randomize_model_weights(model):
        for param in model.parameters():
            init.normal_(param.data, mean=0.0, std=0.02)  # Added seed setting

    randomize_model_weights(random_init_model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "roneneldan/TinyStories-1M",
        revision="8cd14d5",
        cache_dir="./data/",
        padding_side="left",  # Left padding so generate works
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tinystories_model, random_init_model, tokenizer


def integer_to_unary_sequence(n: int) -> list:
    check(ValueError, n < 1, "Integer must be at least 1.")
    return ["1" * i for i in range(1, n + 1)]


def create_dataset_split(
    train_test_split: Union[float, Set[int]], n: int = 100, seed: int = 42
) -> Tuple[List[int], List[int]]:

    data = list(range(1, n + 1))

    if isinstance(train_test_split, float):
        if not (0 <= train_test_split <= 1):
            raise ValueError("float must be between 0 and 1")
        train_data, test_data = sklearn_train_test_split(
            data, test_size=train_test_split, random_state=seed
        )
    elif isinstance(train_test_split, set):
        if not all(isinstance(x, int) and 0 <= x <= 9 for x in train_test_split):
            raise ValueError("integers must be between 0 and 9.")
        train_data = [x for x in data if x % 10 not in train_test_split]
        test_data = [x for x in data if x % 10 in train_test_split]

    return train_data, test_data


class TokenizedDataset(TorchDataset):
    def __init__(self, datapoints, tokenizer, context_window_size=1536):
        self.tokenizer = tokenizer
        self.partial_tokenizer = partial(
            tokenizer,
            padding="max_length",
            max_length=context_window_size,
            truncation=True,
            return_tensors="pt",
        )
        self.context_window_size = context_window_size
        self.prompts = list(map(self._tokenize_prompt, datapoints))
        self.prompt_input_ids, self.prompt_attention_masks = self._properly_stack(
            self.prompts
        )
        self.targets = list(map(self._tokenize_target, datapoints))
        self.target_input_ids, self.target_attention_masks = self._properly_stack(
            self.targets
        )

    def _tokenize_prompt(self, data):
        prompt = f"Please count up in unary, starting at 1 and stopping at {data}:"
        return self.partial_tokenizer(prompt)

    def _tokenize_target(self, data):
        unary_sequence = " ".join(integer_to_unary_sequence(data)) + "<|endoftext|>"
        return self.partial_tokenizer(unary_sequence)

    def _properly_stack(self, data):
        input_ids = torch.vstack([d["input_ids"] for d in data])
        attention_mask = torch.vstack([d["attention_mask"] for d in data])
        return input_ids, attention_mask

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            self.prompt_input_ids[idx],
            self.prompt_attention_masks[idx],
            self.target_input_ids[idx],
            self.target_attention_masks[idx],
        )

    def get_string_prompt(self, idx):
        prompt = self.prompt_input_ids[idx].tolist()
        attention_mask = self.prompt_attention_masks[idx].tolist()
        sos_idx = attention_mask.index(1)
        return self.tokenizer.decode(prompt[sos_idx:])

    def get_string_target(self, idx):
        target = self.target_input_ids[idx].tolist()
        attention_mask = self.target_attention_masks[idx].tolist()
        sos_idx = attention_mask.index(1)
        return self.tokenizer.decode(target[sos_idx:])


def evaluate_model(
    model: PreTrainedModel,
    tokenized_dataset: TokenizedDataset,
    batch_size: int = 4,
):

    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            (
                prompt_input_ids,
                prompt_attention_mask,
                target_input_ids,
                target_attention_mask,
            ) = batch
            # Get model outputs
            outputs = model(
                input_ids=prompt_input_ids, attention_mask=prompt_attention_mask
            )

            # Get logits and predict the tokens
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            # apply mask to predictions
            predictions = predictions * target_attention_mask
            targets = target_input_ids * target_attention_mask

            # Calculate accuracy
            # check where rows are equal
            correct_predictions += (
                torch.all(predictions == targets, dim=-1).sum().item()
            )

            total_predictions += len(targets)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return accuracy


if __name__ == "__main__":
    tinystories_model, random_init_model, tokenizer = load_models()
    dataset = [1, 2, 3, 4]
    tokenized_dataset = TokenizedDataset(dataset, tokenizer, 1536)
    print(evaluate_model(tinystories_model, tokenized_dataset))


# %%
