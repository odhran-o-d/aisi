import lightning as L
from torch import nn, optim
import torch
import wandb
import fire
from dataloader import TokenizedDataset, create_dataset_split
from evaluation import per_batch_calculation
from models import load_models
from torch.utils.data import DataLoader
import torch.nn.functional as F


# define the LightningModule
class LightningTrainer(L.LightningModule):
    def __init__(self, model, learning_rate=2e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        predictions, targets, logits, mask = per_batch_calculation(self.model, batch)
        # mask the logits
        logits = logits * mask.unsqueeze(-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        wandb.log({"train_loss": loss, "batch_idx": batch_idx})
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, targets, logits, mask = per_batch_calculation(self.model, batch)
        # mask the logits
        logits = logits * mask.unsqueeze(-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        wandb.log({"val_loss": loss, "batch_idx": batch_idx})
        accuracy = torch.all(predictions == targets, dim=-1).sum().item() / len(targets)
        wandb.log({"val_accuracy": accuracy, "batch_idx": batch_idx})
        return accuracy

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.learning_rate)
        return optimizer


def train_model(
    learning_rate=2e-4,
    n=100,
    batch_size=10,
    epochs=10,
    seed=42,
    train_test_split=0.2,
):
    wandb.init(project="LLM-training")
    tinystories_model, random_init_model, tokenizer = load_models()
    train_data, test_data = create_dataset_split(train_test_split, n, seed)
    train_dataset = TokenizedDataset(train_data, tokenizer, 1536)
    test_dataset = TokenizedDataset(test_data, tokenizer, 1536)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = LightningTrainer(tinystories_model, learning_rate)
    trainer = L.Trainer(max_epochs=epochs)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    fire.Fire(train_model)
