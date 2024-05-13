from functools import partial
from typing import List, Set, Tuple, Union
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def check(error_type, condition, message):
    if condition:
        raise error_type(message)


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
        self.datapoints = datapoints
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
            self.datapoints[idx],
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
