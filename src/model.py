import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Union
from collections import OrderedDict
import torchvision.models as models

class SteatosisModel(nn.Module):
    """DenseNet121 model adapted for steatosis classification."""
    
    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 2,
        freeze_layers: bool = True
    ):
        """
        Initialize the model.
        
        Args:
            pretrained_path: Path to preprocessed DenseNet121 state dict
            num_classes: Number of output classes (2 for binary, 4 for multi-class)
            freeze_layers: Whether to freeze pretrained layers initially
        """
        super().__init__()
        
        # Initialize base DenseNet121 model
        self.model = models.densenet121(weights=None)
        
        # Load preprocessed weights
        try:
            state_dict = torch.load(pretrained_path)
            self.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            raise
        
        # Store current device
        self.device = next(self.model.parameters()).device
        
        # Modify final classifier
        in_features = self.model.classifier.in_features
        if num_classes == 2:  # Binary classification
            self.model.classifier = nn.Linear(in_features, 1)  # Single output
        else:  # Multi-class
            self.model.classifier = nn.Linear(in_features, num_classes)
        
        # Initialize new layers
        nn.init.xavier_uniform_(self.model.classifier.weight)
        nn.init.zeros_(self.model.classifier.bias)
        
        if freeze_layers:
            self.freeze_pretrained_layers()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def freeze_pretrained_layers(self) -> None:
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_layers(self, num_blocks: Optional[int] = None) -> None:
        """
        Unfreeze a specified number of blocks from the end.
        
        Args:
            num_blocks: Number of dense blocks to unfreeze (None for all)
        """
        # If num_blocks is None, unfreeze all layers
        if num_blocks is None:
            for param in self.model.parameters():
                param.requires_grad = True
            return
        
        # Keep track of blocks we've seen
        blocks_seen = 0
        
        # Iterate through named parameters in reverse
        for name, param in reversed(list(self.model.named_parameters())):
            if 'denseblock' in name:
                if blocks_seen < num_blocks:
                    param.requires_grad = True
                    if 'denselayer1.norm1' in name:  # New block
                        blocks_seen += 1
                else:
                    break
            elif 'classifier' in name:
                param.requires_grad = True
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    @staticmethod
    def create_optimizer(
        params,
        optimizer_type: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> torch.optim.Optimizer:
        """
        Create optimizer for training.
        
        Args:
            params: Model parameters to optimize
            optimizer_type: Type of optimizer ('adam' or 'sgd')
            lr: Learning rate
            weight_decay: Weight decay factor
            
        Returns:
            Configured optimizer
        """
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'plateau',
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            scheduler_type: Type of scheduler ('plateau' or 'step')
            **kwargs: Additional scheduler parameters
            
        Returns:
            Configured scheduler
        """
        if scheduler_type.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # For accuracy/F1
                patience=kwargs.get('patience', 3),
                factor=kwargs.get('factor', 0.1),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 7),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        num_classes: int = 2,
        device: Optional[str] = None
    ) -> 'SteatosisModel':
        """
        Load model from state dict.
        
        Args:
            path: Path to saved model state dict
            num_classes: Number of output classes
            device: Device to load model to ('cuda' or 'cpu')
            
        Returns:
            Loaded model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        model = cls(path, num_classes=num_classes)
        model = model.to(device)
        return model

def get_loss_fn(num_classes: int) -> nn.Module:
    """
    Get appropriate loss function based on number of classes.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Loss function
    """
    if num_classes == 2:
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()