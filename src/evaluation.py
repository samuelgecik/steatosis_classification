from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

class MetricsCalculator:
    """Calculator for model evaluation metrics with GPU acceleration support."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the metrics calculator.
        
        Args:
            device (str): Device to perform calculations on ('cuda' or 'cpu')
        """
        self.device = device
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the specified device."""
        return tensor.to(self.device)
    
    def compute_basic_metrics(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            y_true: Ground truth labels (binary)
            y_pred: Model predictions (probabilities)
            threshold: Classification threshold for probabilities
            
        Returns:
            Dictionary containing computed metrics
        """
        # Move tensors to device and apply threshold
        y_true = self._to_device(y_true)
        y_pred = self._to_device(y_pred)
        y_pred_binary = (y_pred >= threshold).float()
        
        # Compute confusion matrix elements
        tp = torch.sum((y_pred_binary == 1) & (y_true == 1)).float()
        tn = torch.sum((y_pred_binary == 0) & (y_true == 0)).float()
        fp = torch.sum((y_pred_binary == 1) & (y_true == 0)).float()
        fn = torch.sum((y_pred_binary == 0) & (y_true == 1)).float()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0)
        specificity = tn / (tn + fp) if tn + fp > 0 else torch.tensor(0.0)
        precision = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.0)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) \
            if precision + sensitivity > 0 else torch.tensor(0.0)
            
        return {
            'sensitivity': sensitivity.item(),
            'specificity': specificity.item(),
            'precision': precision.item(),
            'accuracy': accuracy.item(),
            'f1_score': f1_score.item()
        }
    
    def compute_multiclass_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for multi-class classification.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (probabilities)
            
        Returns:
            Dictionary containing per-class and macro-averaged metrics
        """
        num_classes = y_pred.shape[1]
        y_true_onehot = F.one_hot(y_true, num_classes)
        y_pred_labels = torch.argmax(y_pred, dim=1)
        
        per_class_metrics = {}
        macro_metrics = defaultdict(float)
        
        # Compute per-class metrics
        for class_idx in range(num_classes):
            binary_true = (y_true == class_idx).float()
            binary_pred = y_pred[:, class_idx]
            
            metrics = self.compute_basic_metrics(binary_true, binary_pred)
            per_class_metrics[f'class_{class_idx}'] = metrics
            
            # Accumulate for macro averaging
            for metric_name, value in metrics.items():
                macro_metrics[f'macro_{metric_name}'] += value
        
        # Compute macro-averages
        for metric_name in macro_metrics:
            macro_metrics[metric_name] /= num_classes
            
        return {
            'per_class': per_class_metrics,
            'macro': dict(macro_metrics)
        }
    
    def compute_roc_auc(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        multi_class: bool = False
    ) -> Dict[str, float]:
        """
        Compute ROC AUC score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (probabilities)
            multi_class: Whether to compute ROC AUC for multi-class
            
        Returns:
            Dictionary containing ROC AUC scores
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        if not multi_class:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            return {'roc_auc': roc_auc}
        
        # Multi-class ROC AUC
        n_classes = y_pred.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Compute per-class ROC AUC
        y_true_onehot = np.eye(n_classes)[y_true]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
            roc_auc[f'class_{i}_roc_auc'] = auc(fpr[i], tpr[i])
        
        # Compute macro-average ROC AUC
        roc_auc['macro_roc_auc'] = sum(
            v for k, v in roc_auc.items() if k != 'macro_roc_auc'
        ) / n_classes
        
        return roc_auc
    
    def plot_roc_curves(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curves and save if path is provided.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (probabilities)
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        if y_pred.shape[1] > 1:  # Multi-class
            n_classes = y_pred.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true.cpu().numpy()]
            y_pred_np = y_pred.cpu().numpy()
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_np[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr, 
                    label=f'Class {i} (AUC = {roc_auc:.2f})'
                )
        else:  # Binary
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            fpr, tpr, _ = roc_curve(y_true_np, y_pred_np)
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                label=f'ROC curve (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix and save if path is provided.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (class indices)
            class_names: Optional list of class names
            save_path: Optional path to save the plot
        """
        # Convert predictions to class indices if they're probabilities
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        
        # Compute confusion matrix
        n_classes = len(class_names) if class_names else \
            max(torch.max(y_true).item(), torch.max(y_pred).item()) + 1
        conf_matrix = torch.zeros((n_classes, n_classes), device=self.device)
        
        for t, p in zip(y_true, y_pred):
            conf_matrix[t.long(), p.long()] += 1
            
        # Convert to percentages
        conf_matrix = conf_matrix.float()
        conf_matrix = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)
        conf_matrix = conf_matrix.cpu().numpy()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, cmap='Blues')
        
        # Add labels
        if class_names:
            plt.xticks(range(n_classes), class_names, rotation=45)
            plt.yticks(range(n_classes), class_names)
        
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, f'{conf_matrix[i, j]:.2%}',
                        ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def generate_report(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a formatted report of the metrics.
        
        Args:
            metrics: Dictionary containing computed metrics
            save_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report = ["# Model Evaluation Report\n"]
        
        # Add timestamp
        from datetime import datetime
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Format metrics
        for category, values in metrics.items():
            report.append(f"\n## {category.title()}\n")
            if isinstance(values, dict):
                for metric, value in values.items():
                    if isinstance(value, (int, float)):
                        report.append(f"- {metric}: {value:.4f}")
                    else:
                        report.append(f"- {metric}: {value}")
            else:
                report.append(f"- Value: {values}")
        
        report_str = "\n".join(report)
        
        if save_path:
            Path(save_path).write_text(report_str)
        
        return report_str