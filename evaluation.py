# %%
import copy
import random
import re
import time
from typing import Any, List, Literal, Set, Tuple, TypedDict, Union
import torch
import wandb
import transformers
import torch

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
    CausalLMOutputWithPast,
)

from pathlib import Path

from dataloader import TokenizedDataset
from models import DummyModel, load_models

Path("./data/").mkdir(parents=True, exist_ok=True)
Path("./cache_dir/").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_batch_calculation(model, batch):
    (
        datapoints,
        prompt_input_ids,
        prompt_attention_mask,
        target_input_ids,
        target_attention_mask,
    ) = batch
    outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
    if type(outputs) == CausalLMOutputWithPast:
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    else:
        predictions = outputs
        logits = None
    # apply mask to predictions
    predictions = predictions * target_attention_mask
    targets = target_input_ids * target_attention_mask

    return predictions, targets, logits, target_attention_mask


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
            predictions, targets, _, _ = per_batch_calculation(model, batch)

            correct_predictions += (
                torch.all(predictions == targets, dim=-1).sum().item()
            )

            total_predictions += len(targets)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return accuracy


if __name__ == "__main__":
    tinystories_model, random_init_model, tokenizer = load_models()
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tokenized_dataset = TokenizedDataset(dataset, tokenizer, 1536)
    print(evaluate_model(tinystories_model, tokenized_dataset))

    dummy_model = DummyModel(
        errors_per_10_datapoints=3,
        tokenizer=tokenizer,
    )
    print(evaluate_model(dummy_model, tokenized_dataset))


# %%
