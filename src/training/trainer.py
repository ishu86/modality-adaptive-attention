"""Training loop for Vision-Text models."""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .scheduler import create_scheduler
from .loss import CaptioningLoss, compute_perplexity
from ..utils.logging import log_metrics
from ..utils.checkpointing import save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer for Vision-Text models.

    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Gradient clipping
    - Learning rate scheduling with warmup
    - Checkpointing
    - W&B logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer (created from config if None)
            scheduler: Learning rate scheduler
            config: Training configuration
            device: Device to use
        """
        self.config = config or {}
        self.training_config = self.config.get("training", {})

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Training parameters
        self.learning_rate = self.training_config.get("learning_rate", 1e-4)
        self.weight_decay = self.training_config.get("weight_decay", 0.01)
        self.warmup_steps = self.training_config.get("warmup_steps", 2000)
        self.total_steps = self.training_config.get("total_steps", 30000)
        self.max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = self.training_config.get(
            "gradient_accumulation_steps", 4
        )
        self.mixed_precision = self.training_config.get("mixed_precision", True)

        # Optimizer
        if optimizer is None:
            optimizer_config = self.config.get("optimizer", {})
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
                eps=optimizer_config.get("eps", 1e-8),
            )
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            scheduler_config = self.config.get("scheduler", {})
            self.scheduler = create_scheduler(
                name=scheduler_config.get("name", "cosine"),
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:
            self.scheduler = scheduler

        # Loss function
        model_config = self.config.get("model", {})
        self.loss_fn = CaptioningLoss(
            vocab_size=model_config.get("vocab_size", 50257),
            label_smoothing=self.training_config.get("label_smoothing", 0.0),
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Checkpointing
        checkpoint_config = self.config.get("checkpoint", {})
        self.save_every = checkpoint_config.get("save_every", 1000)
        self.save_dir = Path(checkpoint_config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        logging_config = self.config.get("logging", {})
        self.log_every = logging_config.get("log_every", 100)

        # Evaluation
        eval_config = self.config.get("evaluation", {})
        self.eval_every = eval_config.get("eval_every", 2000)

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train(
        self,
        num_epochs: Optional[int] = None,
        resume_from: Optional[str] = None,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Run training loop.

        Args:
            num_epochs: Number of epochs (overrides total_steps if set)
            resume_from: Path to checkpoint to resume from
            eval_fn: Custom evaluation function

        Returns:
            Dictionary with final training metrics
        """
        # Resume from checkpoint
        if resume_from:
            self.load(resume_from)

        # Calculate epochs if not specified
        steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        if num_epochs is None:
            num_epochs = max(1, self.total_steps // steps_per_epoch)

        print(f"Training for {num_epochs} epochs ({self.total_steps} steps)")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.train_dataloader.batch_size * self.gradient_accumulation_steps}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Trainable parameters: {self.model.get_num_params():,}")

        # Training metrics
        train_metrics = {
            "train_loss": 0.0,
            "learning_rate": self.learning_rate,
            "throughput": 0.0,
        }

        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True,
            )

            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue

                # Training step
                loss = self.training_step(batch, batch_idx)

                if loss is not None:
                    epoch_loss += loss
                    epoch_steps += 1

                # Update progress bar
                if epoch_steps > 0:
                    pbar.set_postfix({
                        "loss": f"{epoch_loss / epoch_steps:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        "step": self.global_step,
                    })

                # Check if we've reached total steps
                if self.global_step >= self.total_steps:
                    break

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            throughput = epoch_steps * self.train_dataloader.batch_size / epoch_time

            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
            print(f"Average loss: {avg_epoch_loss:.4f}")
            print(f"Throughput: {throughput:.1f} samples/sec")

            # Reset epoch metrics
            epoch_loss = 0.0
            epoch_steps = 0

            if self.global_step >= self.total_steps:
                break

        # Final metrics
        train_metrics["train_loss"] = avg_epoch_loss
        train_metrics["throughput"] = throughput

        return train_metrics

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Optional[float]:
        """
        Single training step.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss value (or None if accumulating)
        """
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass with mixed precision
        with autocast(enabled=self.mixed_precision):
            outputs = self.model(
                input_ids=batch["input_ids"],
                vision_features=batch.get("vision_features"),
                images=batch.get("images"),
                attention_mask=batch.get("full_attention_mask"),
                modality_mask=batch.get("modality_mask"),
                labels=batch["labels"],
            )
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            # Optimizer step
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # Logging
            if self.global_step % self.log_every == 0:
                metrics = {
                    "train/loss": loss.item() * self.gradient_accumulation_steps,
                    "train/perplexity": compute_perplexity(loss * self.gradient_accumulation_steps).item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/global_step": self.global_step,
                }
                log_metrics(metrics, step=self.global_step)

            # Checkpointing
            if self.global_step % self.save_every == 0:
                self.save(f"checkpoint-{self.global_step}")

            # Evaluation
            if self.val_dataloader and self.global_step % self.eval_every == 0:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save("best")

            return loss.item() * self.gradient_accumulation_steps

        return None

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            if batch is None:
                continue

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    vision_features=batch.get("vision_features"),
                    images=batch.get("images"),
                    attention_mask=batch.get("full_attention_mask"),
                    modality_mask=batch.get("modality_mask"),
                    labels=batch["labels"],
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = compute_perplexity(torch.tensor(avg_loss)).item()

        print(f"\nValidation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

        log_metrics({
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
        }, step=self.global_step)

        self.model.train()
        return avg_loss

    def save(self, name: str):
        """Save checkpoint."""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.config,
            path=self.save_dir / name,
        )
        print(f"Saved checkpoint: {name}")

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        print(f"Loaded checkpoint from step {self.global_step}")
