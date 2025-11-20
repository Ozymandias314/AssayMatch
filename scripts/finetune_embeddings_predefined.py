import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import os
import argparse
import sys

# --- Dataset Class for Pre-defined Triplets ---

class PredefinedTripletDataset(Dataset):
    """
    Dataset for loading pre-defined training or validation/test triplets.
    It takes a path to a .npy file where each row is [anchor, positive, negative].
    """
    def __init__(self, triplets_file, dataset_name="data"):
        """
        Args:
            triplets_file (str): Path to the .npy file containing the triplets.
            dataset_name (str): Name for logging purposes (e.g., "training", "validation").
        """
        super().__init__()
        print(f"Loading {dataset_name} triplets from {triplets_file}...")
        if not os.path.exists(triplets_file):
            raise FileNotFoundError(f"{dataset_name.capitalize()} triplets file not found: {triplets_file}")
        
        self.triplets = np.load(triplets_file)
        
        if self.triplets.size == 0:
            raise ValueError(f"The {dataset_name} triplets file is empty: {triplets_file}")
            
        print(f"Loaded {len(self.triplets)} {dataset_name} triplets.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Return indices: anchor, positive, negative
        return self.triplets[idx, 0], self.triplets[idx, 1], self.triplets[idx, 2]


# --- Model Definition ---

class ProjectionMLP(nn.Module):
    """A simple MLP projection head with L2 normalization."""
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(1, input_dim // 2) # Default hidden size

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        print(f"Initialized ProjectionMLP: Input={input_dim}, Hidden={hidden_dim}, Output={output_dim}")

    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# --- PyTorch Lightning Module ---

class TripletFineTuner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        print("Initializing LightningModule...")
        
        print(f"Memory-mapping embeddings from: {self.hparams.embedding_file}")
        if not os.path.exists(self.hparams.embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {self.hparams.embedding_file}")

        try:
            embeddings_memmap = np.load(self.hparams.embedding_file, mmap_mode='r')
            self.embedding_dim = embeddings_memmap.shape[1]
            self.num_embeddings = embeddings_memmap.shape[0]
        except Exception as e:
            print(f"Error loading embedding file shape: {e}")
            raise

        print(f"Embeddings shape: ({self.num_embeddings}, {self.embedding_dim})")

        self.mlp = ProjectionMLP(
            input_dim=self.embedding_dim,
            output_dim=self.hparams.projection_dim,
            hidden_dim=int(self.embedding_dim * self.hparams.hidden_dim_ratio)
        )

        self.triplet_loss = nn.TripletMarginLoss(
            margin=self.hparams.margin,
            p=2, # Euclidean distance
            eps=1e-7
        )
        print(f"Using TripletMarginLoss with margin={self.hparams.margin}")

        self.validation_step_outputs = []

    def _get_embeddings(self, indices):
        """Fetches embeddings by indices from the memory-mapped file."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        indices = np.asarray(indices)

        try:
            # Re-open memmap in each call for worker safety
            embeddings_memmap = np.load(self.hparams.embedding_file, mmap_mode='r')
            embeds = torch.from_numpy(embeddings_memmap[indices].copy()).float()
            return embeds.to(self.device)
        except IndexError as e:
            print(f"IndexError in _get_embeddings: {e}. Indices: {indices}")
            raise
        except Exception as e:
            print(f"Error loading embeddings slice: {e}")
            raise

    def forward(self, indices):
        """Project embeddings corresponding to indices."""
        original_embeddings = self._get_embeddings(indices)
        projected_embeddings = self.mlp(original_embeddings)
        return projected_embeddings

    def _process_batch(self, batch):
        """Shared logic for processing a batch in both training and validation."""
        anchor_idx, positive_idx, negative_idx = batch
        
        anchor_idx_t = torch.as_tensor(anchor_idx, device=self.device)
        positive_idx_t = torch.as_tensor(positive_idx, device=self.device)
        negative_idx_t = torch.as_tensor(negative_idx, device=self.device)

        unique_indices, inverse_indices = torch.unique(
            torch.cat([anchor_idx_t, positive_idx_t, negative_idx_t]),
            return_inverse=True
        )

        # The forward pass projects the embeddings for the unique indices
        projected_embeddings = self.forward(unique_indices)

        # Reconstruct the batch of embeddings using the inverse indices
        anchor_embeds = projected_embeddings[inverse_indices[:anchor_idx_t.size(0)]]
        positive_embeds = projected_embeddings[inverse_indices[anchor_idx_t.size(0):anchor_idx_t.size(0)*2]]
        negative_embeds = projected_embeddings[inverse_indices[anchor_idx_t.size(0)*2:]]
        
        return anchor_embeds, positive_embeds, negative_embeds

    def training_step(self, batch, batch_idx):
        try:
            anchor_embeds, positive_embeds, negative_embeds = self._process_batch(batch)
            loss = self.triplet_loss(anchor_embeds, positive_embeds, negative_embeds)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        except Exception as e:
            print(f"Error in training_step batch {batch_idx}: {e}")
            # Log the problematic indices for debugging
            anchor_idx, positive_idx, negative_idx = batch
            print(f"Anchor indices: {anchor_idx}")
            print(f"Positive indices: {positive_idx}")
            print(f"Negative indices: {negative_idx}")
            raise

    def validation_step(self, batch, batch_idx):
        try:
            with torch.no_grad():
                anchor_embeds, positive_embeds, negative_embeds = self._process_batch(batch)

            dist_ap = torch.pairwise_distance(anchor_embeds, positive_embeds, p=2)
            dist_an = torch.pairwise_distance(anchor_embeds, negative_embeds, p=2)

            correct = (dist_ap < dist_an).float()
            accuracy = correct.mean()

            self.validation_step_outputs.append(accuracy)
            self.log('val_accuracy_step', accuracy, on_step=True, on_epoch=False)
            return accuracy
        except Exception as e:
            print(f"Error in validation_step batch {batch_idx}: {e}")
            anchor_idx, positive_idx, negative_idx = batch
            print(f"Anchor indices: {anchor_idx}")
            print(f"Positive indices: {positive_idx}")
            print(f"Negative indices: {negative_idx}")
            return torch.tensor(float('nan'), device=self.device) # Return NaN to indicate error

    def on_validation_epoch_end(self):
        valid_outputs = [out for out in self.validation_step_outputs if not torch.isnan(out)]

        if not valid_outputs:
            print("Warning: No valid validation outputs recorded for this epoch.")
            avg_accuracy = torch.tensor(0.0, device=self.device)
        else:
            avg_accuracy = torch.stack(valid_outputs).mean()

        self.log('val_accuracy_epoch', avg_accuracy, on_epoch=True, prog_bar=True, logger=True)
        print(f"\nEpoch {self.current_epoch}: Validation Accuracy = {avg_accuracy:.4f}")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune embeddings using Triplet Loss with pre-defined triplets.")

    # Data paths
    parser.add_argument('--embedding_file', type=str, default='embeddings.npy',
                        help='Path to embeddings .npy file (default: embeddings.npy)')
    parser.add_argument('--train_triplets_file', type=str, required=True,
                        help='Path to pre-defined training triplets .npy file (Required)')
    parser.add_argument('--validation_triplets_file', type=str, default='validation_triplets.npy',
                        help='Path to validation triplets .npy file (default: validation_triplets.npy)')

    # Model Hyperparameters
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Output dimension of the MLP (default: 128)')
    parser.add_argument('--hidden_dim_ratio', type=float, default=0.5,
                        help='Hidden dim ratio for MLP relative to input embedding dim (default: 0.5)')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=4e-5,
                        help='Learning rate (default: 4e-5)')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Triplet loss margin (default: 0.5)')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping (default: 20)')

    # System/Infrastructure
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers (default: 4)')
    parser.add_argument('--accelerator', type=str, default='auto', choices=['cpu', 'gpu', 'tpu', 'mps', 'auto'],
                        help='PyTorch Lightning accelerator (default: auto)')
    parser.add_argument('--output_dir', type=str, default='triplet_finetune_output',
                        help='Directory to save checkpoints and logs (default: triplet_finetune_output)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Optional random seed for reproducibility')
    parser.add_argument('--results_path', type=str, required=False, help='Path to save results.') 
    # Logging
    parser.add_argument('--experiment_name', type=str, required=False, help='Name for wandb experiment.')

    parser.add_argument('--model_type', type=str, default='', help='Type of model architecture trak was trained with.')

    args = parser.parse_args()
    return args

# --- Main Execution Block ---

if __name__ == '__main__':
    args = parse_args()

    if args.random_seed is not None:
        print(f"Setting random seed to: {args.random_seed}")
        pl.seed_everything(args.random_seed, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup Data ---
    print("Setting up datasets...")
    try:
        train_dataset = PredefinedTripletDataset(
            triplets_file=args.train_triplets_file,
            dataset_name="training"
        )
        val_dataset = PredefinedTripletDataset(
            triplets_file=args.validation_triplets_file,
            dataset_name="validation"
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error initializing datasets: {e}")
        sys.exit(1)

    # Worker init function for reproducibility
    def seed_worker(worker_id):
        if args.random_seed is not None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=seed_worker if args.random_seed is not None else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=seed_worker if args.random_seed is not None else None
    )

    # --- Setup Model ---
    print("Initializing model...")
    model = TripletFineTuner(args)

    # --- Setup Callbacks & Logger ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='best-model-{epoch}-{val_accuracy_epoch:.4f}' + args.model_type,
        save_top_k=1,
        verbose=True,
        monitor='val_accuracy_epoch',
        mode='max'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy_epoch',
        patience=args.patience,
        verbose=True,
        mode='max'
    )

    wandb_logger_name = args.experiment_name or f"proj_dim={args.projection_dim}_lr={args.learning_rate}_margin={args.margin}"
    wandb_logger = WandbLogger(
        name=wandb_logger_name,
        project='improved_finetuning', # Or your desired project name
        log_model=True
    )

    # --- Setup Trainer ---
    print("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=20,
        deterministic=True if args.random_seed is not None else False,
        logger=wandb_logger
    )

    # --- Start Training ---
    print("Starting training...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nTraining finished.")
    if checkpoint_callback.best_model_path:
        print(f"Best model checkpoint saved in: {checkpoint_callback.best_model_path}")
    else:
        print("No checkpoint was saved.")