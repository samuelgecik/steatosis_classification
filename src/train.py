import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
from datetime import datetime
import logging
from tqdm import tqdm

from src.model import SteatosisModel, get_loss_fn
from src.data import create_dataloaders
from src.evaluation import MetricsCalculator

class Trainer:
    """Handles model training with phased fine-tuning strategy."""
    
    def __init__(
        self,
        model: SteatosisModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: torch.nn.Module,
        metrics_calculator: MetricsCalculator,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: Optional[str] = None
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.metrics_calculator = metrics_calculator
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path('training_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'best_metric': 0.0
        }
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                if output.shape[1] == 1:  # Binary classification
                    output = output.squeeze()
                    target = target.float()
                
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate model and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if output.shape[1] == 1:  # Binary classification
                    output = output.squeeze()
                    target = target.float()
                    
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                all_outputs.append(output)
                all_targets.append(target)
        
        # Concatenate all predictions and targets
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        
        # Apply activation function for predictions
        if outputs.shape[-1] == 1:  # Binary
            outputs = torch.sigmoid(outputs)
        else:  # Multi-class
            outputs = torch.softmax(outputs, dim=1)
        
        # Calculate metrics
        metrics = self.metrics_calculator.compute_basic_metrics(
            targets,
            outputs
        )
        metrics.update(
            self.metrics_calculator.compute_roc_auc(
                targets,
                outputs,
                multi_class=outputs.shape[-1] > 1
            )
        )
        
        return total_loss / len(self.val_loader), metrics
    
    def train(
        self,
        num_epochs: List[int],
        unfreeze_blocks: List[Optional[int]],
        learning_rates: List[float],
        early_stopping_patience: int = 5
    ) -> None:
        """
        Train model with phased fine-tuning.
        
        Args:
            num_epochs: List of epochs for each phase
            unfreeze_blocks: List of blocks to unfreeze in each phase
            learning_rates: List of learning rates for each phase
            early_stopping_patience: Number of epochs to wait for improvement
        """
        assert len(num_epochs) == len(unfreeze_blocks) == len(learning_rates), \
            "Number of phases must match"
        
        # Track best model
        best_metric = 0.0
        patience_counter = 0
        
        for phase, (epochs, blocks, lr) in enumerate(zip(
            num_epochs, unfreeze_blocks, learning_rates
        )):
            logging.info(f"\nStarting Phase {phase + 1}")
            logging.info(f"Unfreezing {blocks if blocks else 'all'} blocks")
            logging.info(f"Learning rate: {lr}")
            
            # Unfreeze specified blocks
            self.model.unfreeze_layers(blocks)
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            for epoch in range(epochs):
                logging.info(f"\nEpoch {epoch + 1}/{epochs}")
                
                # Train and validate
                train_loss = self.train_epoch()
                val_loss, metrics = self.validate()
                
                # Update learning rate scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics['f1_score'])
                else:
                    self.scheduler.step()
                
                # Log metrics
                logging.info(
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val F1: {metrics['f1_score']:.4f}, "
                    f"Val AUC: {metrics['roc_auc']:.4f}"
                )
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics'].append(metrics)
                
                # Save best model
                if metrics['f1_score'] > best_metric:
                    best_metric = metrics['f1_score']
                    self.save_checkpoint('best_model.pt', metrics)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logging.info(
                        f"Early stopping triggered after {patience_counter} epochs "
                        "without improvement"
                    )
                    break
            
            # Save phase checkpoint
            self.save_checkpoint(f'phase_{phase + 1}_model.pt', metrics)
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, filename: str, metrics: Dict) -> None:
        """Save model checkpoint with metrics."""
        save_path = self.output_dir / filename
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint to {save_path}")
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logging.info(f"Saved training history to {history_path}")

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train steatosis classifier')
    parser.add_argument('--model', type=str, default='Models/DenseNet121.pt',
                      help='Path to pretrained model')
    parser.add_argument('--data', type=str, default='DataSet',
                      help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='training_output',
                      help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--binary', action='store_true',
                      help='Use binary classification')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        binary=args.binary
    )
    
    # Initialize model
    num_classes = 2 if args.binary else 4
    model = SteatosisModel(
        pretrained_path=args.model,
        num_classes=num_classes,
        freeze_layers=True
    ).to(device)
    
    # Setup training components
    optimizer = model.create_optimizer(
        model.get_trainable_params(),
        optimizer_type='adam',
        lr=1e-3
    )
    scheduler = model.create_scheduler(optimizer, scheduler_type='plateau')
    loss_fn = get_loss_fn(num_classes)
    metrics_calculator = MetricsCalculator(device=device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics_calculator=metrics_calculator,
        output_dir=output_dir
    )
    
    # Training phases
    num_epochs = [10, 15, 20]  # Epochs per phase
    unfreeze_blocks = [None, 2, None]  # Blocks to unfreeze per phase
    learning_rates = [1e-3, 1e-4, 1e-5]  # Learning rates per phase
    
    # Start training
    trainer.train(
        num_epochs=num_epochs,
        unfreeze_blocks=unfreeze_blocks,
        learning_rates=learning_rates,
        early_stopping_patience=5
    )

if __name__ == '__main__':
    main()