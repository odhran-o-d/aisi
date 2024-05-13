import torch
import transformers
import torch.nn.init as init
from transformers import (
    GenerationConfig,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from torch import FloatTensor, Tensor

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)

from dataloader import integer_to_unary_sequence


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


class DummyModel(GPTNeoForCausalLM):
    """Dummy model that can do unary counting completion."""

    def __init__(
        self,
        config: GPTNeoConfig = transformers.AutoConfig.from_pretrained(
            "roneneldan/TinyStories-1M"
        ),
        errors_per_10_datapoints: int = 0,
        tokenizer: PreTrainedTokenizerBase = None,
    ) -> None:
        super().__init__(config)
        self.errors_per_10_datapoints = errors_per_10_datapoints
        self.tokenizer = tokenizer
        self.batch_counter = 0
        if self.errors_per_10_datapoints < 0 or self.errors_per_10_datapoints > 10:
            raise ValueError("errors_per_10_datapoints must be between 0 and 10")
        """Initialize the model."""

    def forward(
        self,
        input_ids: Tensor | None = None,
        past_key_values: tuple[FloatTensor] | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions:

        correct_tokens = []
        for input, attn in zip(input_ids, attention_mask):
            sos_idx = attn.tolist().index(1)
            unary_sequence = input[sos_idx:].tolist()
            unary_string = self.tokenizer.decode(unary_sequence)
            unary_integer = int(unary_string.split(" ")[-1][:-1])
            correct_text = (
                " ".join(integer_to_unary_sequence(unary_integer)) + "<|endoftext|>"
            )
            correct_token = self.tokenizer(correct_text).input_ids

            correct_token = torch.cat(
                [
                    torch.full(
                        (input_ids.shape[-1] - len(correct_token),),
                        self.tokenizer.eos_token_id,
                    ),
                    torch.tensor(correct_token),
                ]
            )
            if self.batch_counter % 10 < self.errors_per_10_datapoints:
                correct_tokens.append(torch.zeros_like(correct_token))
            else:
                correct_tokens.append(correct_token)

            self.batch_counter += 1
            if self.batch_counter == 10:
                self.batch_counter = 0

        return torch.vstack(correct_tokens)
