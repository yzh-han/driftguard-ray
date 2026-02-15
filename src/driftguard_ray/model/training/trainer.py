"""Minimal training utilities for PyTorch models."""

from dataclasses import dataclass
import os
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F

from driftguard_ray.config import get_logger

logger = get_logger("trainer")

@dataclass
class TrainConfig:
    """Configuration for the training loop.

    Attributes:
        epochs: Number of training epochs.
        device: Device string or torch.device to run on.
        amp: Enable automatic mixed precision if on CUDA.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        grad_clip: Max norm for gradient clipping; None disables clipping.
        accumulate_steps: Number of steps to accumulate gradients.
        early_stop: Enable early stopping based on validation loss.
        early_stop_patience: Epochs to wait without improvement.
        early_stop_min_delta: Minimum improvement to reset patience.
    """
    epochs: int = 10
    device: str | torch.device | None = "cuda" if torch.cuda.is_available() else "cpu"

    # automatic mixed precision
    amp: bool = True if torch.cuda.is_available() else False

    # optimizer params
    lr: float = 1e-3
    weight_decay: float = 0.01

    grad_clip: Optional[float] = None   
    accumulate_steps: int = 1           # 累计梯度步数

    
        # early stopping
    early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    def __post_init__(self) -> None:
        self.amp = (
            True
            if "cuda" in str(self.device).lower()
            and torch.cuda.is_available()
            else False
        )
    
    # record checkpoint name
    cp_name: str = "init.pth"

@dataclass
class EpochMetrics:
    """Aggregated metrics for a single epoch.

    Attributes:
        loss: Mean loss for the epoch.
        accuracy: Mean accuracy for the epoch, if available.
        num_samples: Number of samples processed.
    """

    loss: float = 0.0
    accuracy: float = 0.0
    num_samples: int = 0


class AverageMeter:
    """Running average tracker.

    Attributes:
        total: Sum of values.
        count: Number of items.
    """

    def __init__(self) -> None:
        """Initialize the meter."""
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        """Update the meter.

        Args:
            value: Value to add.
            n: Number of items represented by the value.
        """

        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Return the current average."""
        return self.total / self.count if self.count else 0.0


class Trainer:
    """Simple trainer with optional validation and AMP.

    Attributes:
        model: Model to train.
        optimizer: AdamW optimizer instance.
        loss_fn: Loss function callable.
        config: Training configuration.
        device: Resolved torch.device for execution.
        scaler: GradScaler for AMP if enabled and on CUDA.
        metric_fn: Optional metric function for predictions and targets.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.CrossEntropyLoss(),
        config: Optional[TrainConfig] = None,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            loss_fn: Loss function.
            config: Training configuration.
            metric_fn: Optional metric for accuracy-like values.
        """

        self.model = model
        self.loss_fn = loss_fn
        self.config = config or TrainConfig()
        self.metric_fn = metric_fn or self._default_accuracy

        self.device = torch.device(
            self.config.device
            or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scaler: Optional[GradScaler] = (
            GradScaler("cuda")
            if self.config.amp and self.device.type == "cuda"
            else None
        )
        self._cp_name = self.config.cp_name

    def save(self) -> None:
        """Save the model state dict."""
        os.makedirs("cp", exist_ok=True)
        torch.save(self.model.state_dict(), f"cp/{self._cp_name}.pth")
        logger.info(f"Saved model weights to cp/{self._cp_name}.pth")

    def load(self) -> None:
        """Load the model state dict."""
        os.makedirs("cp", exist_ok=True)
        self.model.load_state_dict(
            torch.load(f"cp/{self._cp_name}.pth", map_location=self.device)
        )
        logger.info(f"Loaded model weights from cp/{self._cp_name}.pth")

    def inference(
        self, 
        test_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[EpochMetrics, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._run_epoch(test_loader, training=False)

    def fit(
        self,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        val_loader: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> list[dict[str, float]]:
        """Run training with optional validation.

        Args:
            train_loader: Iterable of (inputs, targets) for training.
            val_loader: Optional iterable for validation.

        Returns:
            History list with per-epoch metrics.
        """

        history: list[dict[str, float]] = [] # epoch -> metrics
        best_val_loss: float | None = None
        no_improve = 0

        for epoch in range(self.config.epochs):
            train_metrics, _, _, _ = self._run_epoch(train_loader, training=True)
            record = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
            }
            if train_metrics.accuracy is not None:
                record["train_accuracy"] = train_metrics.accuracy

            val_loss: float | None = None
            if val_loader is not None:
                val_metrics, _, _, _ = self._run_epoch(val_loader, training=False)
                val_loss = val_metrics.loss
                record["val_loss"] = val_metrics.loss
                if val_metrics.accuracy is not None:
                    record["val_accuracy"] = val_metrics.accuracy
            history.append(record)
            
            logger.debug(f"Epoch {epoch+1}/{self.config.epochs}: {record}")
            if self.config.early_stop and val_loss is not None:
                if best_val_loss is None or (
                    best_val_loss - val_loss > self.config.early_stop_min_delta
                ):
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.config.early_stop_patience:
                        logger.debug(
                            "Early stopping at epoch %s (val_loss=%.6f).",
                            epoch + 1,
                            val_loss,
                        )
                        break
        return history

    def _run_epoch(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        training: bool,
    ) -> Tuple[EpochMetrics, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a training or evaluation epoch.

        Args:
            loader: Iterable of (inputs, targets).
            training: Whether to run backprop and optimization.

        Returns:
            Aggregated metrics for the epoch along with l1_w and l2_w tensors.
        """

        self.model.train(training)
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        l1_w_list = []
        l2_w_list = []
        out_list = []

        self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            for step, batch in enumerate(loader, start=1):
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.scaler is not None,
                ):
                    out, l1_w, l2_w = self.model(inputs)
                    loss = self.loss_fn(out, targets)

                # 记录 loss
                batch_size = targets.shape[0]
                loss_meter.update(loss.item(), batch_size)

                # 记录 accuracy
                metric_value = self.metric_fn(out, targets)
                if metric_value is not None:
                    acc_meter.update(metric_value, batch_size)

                # 记录 l1_w, l2_w, softs
                if not training:
                    l1_w_list.append(l1_w)
                    l2_w_list.append(l2_w)
                    out_list.append(out)

                if training:
                    self._backward_step(loss, step)

            if training and step % self.config.accumulate_steps != 0:
                self._optimizer_step()

        accuracy = acc_meter.avg if acc_meter.count else 0.0
        l1_w = torch.concat(l1_w_list) if l1_w_list else torch.tensor([])
        l2_w = torch.concat(l2_w_list) if l2_w_list else torch.tensor([])
        X = torch.concat(out_list) if out_list else torch.tensor([])
        return (
            EpochMetrics(
                loss=loss_meter.avg, accuracy=accuracy, num_samples=loss_meter.count
            ),
            l1_w,
            l2_w,
            F.softmax(X, dim=-1)
        )

    def _backward_step(self, loss: torch.Tensor, step: int) -> None:
        """Run backward pass with optional AMP and gradient accumulation.

        Args:
            loss: Loss tensor for the current batch.
            step: Current step index (1-based).
        """

        loss = loss / self.config.accumulate_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % self.config.accumulate_steps != 0:
            return

        self._optimizer_step()

    def _optimizer_step(self) -> None:
        """Apply optimizer step with optional AMP and gradient clipping."""

        if self.config.grad_clip is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _default_accuracy(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Optional[float]:
        """Compute accuracy for classification logits.

        Args:
            outputs: Model outputs (logits).
            targets: Ground-truth labels.

        Returns:
            Accuracy in [0, 1] if shapes are compatible, otherwise None.
        """

        if outputs.dim() < 2 or targets.dim() != 1:
            return None
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.numel()
