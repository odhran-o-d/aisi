import unittest

from datasets import DatasetDict
import transformers
import torch

from dataloader import TokenizedDataset, create_dataset_split, integer_to_unary_sequence
from evaluation import evaluate_model
from models import DummyModel, load_models


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


class TestTokenizeDatasetAndEvaluation(unittest.TestCase):

    def test_tokenize_dataset(self):
        tinystories_model, random_init_model, tokenizer = load_models()
        dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tokenized_dataset = TokenizedDataset(dataset, tokenizer, 1536)
        self.assertEqual(len(tokenized_dataset), 10)

        self.assertEqual(
            tokenized_dataset.get_string_prompt(0),
            "Please count up in unary, starting at 1 and stopping at 1:",
        )

        self.assertEqual(
            tokenized_dataset.get_string_target(2), "1 11 111<|endoftext|>"
        )

        dummy_model = DummyModel(tokenizer=tokenizer, errors_per_10_datapoints=2)
        self.assertEqual(evaluate_model(dummy_model, tokenized_dataset), 0.8)

        dummy_model = DummyModel(tokenizer=tokenizer, errors_per_10_datapoints=10)
        self.assertEqual(evaluate_model(dummy_model, tokenized_dataset), 0.0)

        dummy_model = DummyModel(tokenizer=tokenizer, errors_per_10_datapoints=0)
        self.assertEqual(evaluate_model(dummy_model, tokenized_dataset), 1.0)


if __name__ == "__main__":
    unittest.main()
