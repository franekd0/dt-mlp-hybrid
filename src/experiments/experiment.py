import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import MLP
from src.utils.data_utils import load_wine_dataset
from experiment_result import ExperimentResult


class Experiment:
    def __init__(self, name, model_type, dataset_name, config):
        self.name = name
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result = ExperimentResult(
            experiment_name=name,
            model_type=model_type,
            dataset_name=dataset_name,
            config=config
        )

    def run(self):
        self._load_data()
        self._build_model()
        self._build_optimizer()
        self._train()
        self._finalize()
        return self.result

    def _load_data(self):
        if self.dataset_name != "wine":
            raise NotImplementedError

        X_train, X_val, _, y_train, y_val, _ = load_wine_dataset(
            test_size=0.2,
            val_size=0.25,
            random_state=42
        )

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.long)

        self.train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.config.get("batch_size", 32),
            shuffle=False
        )

    def _build_model(self):
        input_dim = self.X_train.shape[1]
        num_classes = len(torch.unique(self.y_train))

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=self.config["hidden_dim"],
            embedding_dim=self.config["output_dim"],
            num_layers=self.config["num_layers"],
            num_classes=num_classes
        ).to(self.device)

    def _build_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"]
        )

    def _train(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        epochs = self.config["epochs"]

        for epoch in range(epochs):
            self._train_epoch()
            self._validate_epoch(epoch)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.train_losses.append(total_loss / len(self.train_loader))

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total

        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        print(
            f"[{self.name}] Epoch {epoch+1:02d} | "
            f"train loss: {self.train_losses[-1]:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

    def _finalize(self):
        self.result.history["train_loss"] = self.train_losses
        self.result.history["val_loss"] = self.val_losses
        self.result.history["val_accuracy"] = self.val_accuracies

        self.result.metrics["final_val_accuracy"] = self.val_accuracies[-1]
        self.result.artifacts["model"] = self.model

exp = Experiment(
    name="mlp_wine_depth3",
    model_type="mlp",
    dataset_name="wine",
    config={
        "hidden_dim": 128,
        "output_dim": 64,
        "num_layers": 3,
        "lr": 1e-3,
        "epochs": 30,
        "batch_size": 32
    }
)

result = exp.run()