import unittest
from project import (
    # create_dataset,
    create_dataset_split,
    # get_dataset_dict,
    integer_to_unary_sequence,
    # tokenize_dataset,
    TokenizedDataset,
)
from datasets import DatasetDict
import transformers
import torch


class TestCreateDatasetSplit(unittest.TestCase):
    def test_create_dataset_split_float(self):
        train_data, test_data = create_dataset_split(0.2, 100, 42)
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(test_data), 20)

    def test_create_dataset_split_set(self):
        train_data, test_data = create_dataset_split({1, 2, 3}, 100, 42)
        expected_test_data = [x for x in range(1, 101) if x % 10 in {1, 2, 3}]
        expected_train_data = [x for x in range(1, 101) if x % 10 not in {1, 2, 3}]
        self.assertEqual(train_data, expected_train_data)
        self.assertEqual(test_data, expected_test_data)

    def test_create_dataset_split_invalid_float(self):
        with self.assertRaises(ValueError):
            create_dataset_split(1.5, 100, 42)  # Invalid float

    def test_create_dataset_split_invalid_set(self):
        with self.assertRaises(ValueError):
            create_dataset_split({10, 11}, 100, 42)  # Invalid set elements


class TestIntegerToUnarySequence(unittest.TestCase):
    def test_integer_to_unary_sequence(self):
        # Test normal input
        self.assertEqual(
            integer_to_unary_sequence(5), ["1", "11", "111", "1111", "11111"]
        )
        # Test minimum input
        self.assertEqual(integer_to_unary_sequence(1), ["1"])
        # Test value error for zero
        with self.assertRaises(ValueError):
            integer_to_unary_sequence(0)
        # Test value error for negative
        with self.assertRaises(ValueError):
            integer_to_unary_sequence(-1)


class TestTokenizeDataset(unittest.TestCase):
    def test_tokenize_dataset(self):
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        dataset = [1, 2, 3, 4, 5]
        tokenized_dataset = TokenizedDataset(dataset, tokenizer, 1536)
        self.assertEqual(len(tokenized_dataset), 5)

        # get four items and check if they have the right shape
        for i in range(4):
            item = tokenized_dataset[i]
            self.assertEqual(item["input_ids"].shape, torch.Size([1536]))
            self.assertEqual(item["attention_mask"].shape, torch.Size([1536]))


if __name__ == "__main__":
    unittest.main()
