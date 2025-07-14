"""
Model Evaluation
================

Centralized evaluation functionality for ML forecasting models.
Includes accuracy metrics, confusion matrices, and reliability analysis.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import seaborn as sns

from .config import MLConfig


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""
    
    def __init__(self, config: MLConfig):
        self.config = config
    
    def evaluate(self, model: nn.Module, dataset, tag: str = "Test") -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model to evaluate
            dataset: Dataset to evaluate on
            tag: Label for evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"📊 Evaluating model on {tag} set...")
        
        # Get predictions and ground truth
        predictions, probabilities, actual_labels = self._get_predictions(model, dataset)
        
        # Calculate metrics
        results = {
            'tag': tag,
            'accuracy': self._calculate_accuracy(predictions, actual_labels),
            'confusion_matrix': self._calculate_confusion_matrix(predictions, actual_labels),
            'classification_report': self._get_classification_report(predictions, actual_labels),
            'reliability_stats': self._calculate_reliability(probabilities, actual_labels),
            'prediction_distribution': self._analyze_prediction_distribution(predictions),
            'calibration_metrics': self._calculate_calibration(probabilities, actual_labels)
        }
        
        # Print summary
        self._print_evaluation_summary(results)
        
        # Generate plots if enabled
        if self.config.plot_reliability:
            self._generate_evaluation_plots(probabilities, actual_labels, predictions, tag)
        
        return results
    
    def _get_predictions(self, model: nn.Module, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and probabilities."""
        model.eval()
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.config.device)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    # For SimpleModel
                    probs = model.predict_proba(X_batch)
                elif hasattr(model, 'get_logits'):
                    # For MLPClassifier
                    logits = model.get_logits(X_batch)
                    probs = torch.softmax(logits, dim=1)
                else:
                    # Default forward pass
                    probs = model(X_batch)
                
                probs = probs.cpu().numpy()
                predictions = probs.argmax(axis=1)
                
                all_predictions.append(predictions)
                all_probabilities.append(probs)
                all_labels.append(y_batch.numpy())
        
        predictions = np.concatenate(all_predictions)
        probabilities = np.vstack(all_probabilities)
        actual_labels = np.concatenate(all_labels)
        
        return predictions, probabilities, actual_labels
    
    def _calculate_accuracy(self, predictions: np.ndarray, actual_labels: np.ndarray) -> float:
        """Calculate overall accuracy."""
        return accuracy_score(actual_labels, predictions)
    
    def _calculate_confusion_matrix(self, predictions: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        labels = np.arange(self.config.n_quantiles)
        return confusion_matrix(actual_labels, predictions, labels=labels)
    
    def _get_classification_report(self, predictions: np.ndarray, actual_labels: np.ndarray) -> str:
        """Get detailed classification report."""
        # Get actual classes present in the data
        all_classes = np.union1d(actual_labels, predictions)
        target_names = [f"Quantile_{i}" for i in all_classes]
        
        # Use only the classes that actually exist
        return classification_report(actual_labels, predictions, 
                                   labels=all_classes, target_names=target_names)
    
    def _calculate_reliability(self, probabilities: np.ndarray, actual_labels: np.ndarray) -> Dict:
        """Calculate reliability/calibration statistics."""
        reliability_stats = {}
        
        for class_idx in range(self.config.n_quantiles):
            # Binary calibration for each class
            binary_labels = (actual_labels == class_idx).astype(int)
            class_probs = probabilities[:, class_idx]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    binary_labels, class_probs, n_bins=10
                )
                
                # Calculate Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = binary_labels[in_bin].mean()
                        avg_confidence_in_bin = class_probs[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                reliability_stats[f'class_{class_idx}'] = {
                    'ece': ece,
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            except Exception as e:
                print(f"⚠️  Warning: Could not calculate reliability for class {class_idx}: {e}")
                reliability_stats[f'class_{class_idx}'] = {'ece': np.nan}
        
        # Overall ECE
        overall_ece = np.mean([stats.get('ece', np.nan) for stats in reliability_stats.values()])
        reliability_stats['overall_ece'] = overall_ece
        
        return reliability_stats
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray) -> Dict:
        """Analyze distribution of predictions."""
        unique, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        distribution = {}
        for class_idx in range(self.config.n_quantiles):
            count = counts[unique == class_idx][0] if class_idx in unique else 0
            distribution[f'quantile_{class_idx}'] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        return distribution
    
    def _calculate_calibration(self, probabilities: np.ndarray, actual_labels: np.ndarray) -> Dict:
        """Calculate calibration metrics."""
        calibration_metrics = {}
        
        # Brier Score
        one_hot_labels = np.eye(self.config.n_quantiles)[actual_labels]
        brier_score = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
        calibration_metrics['brier_score'] = float(brier_score)
        
        # Log Loss
        epsilon = 1e-15  # To avoid log(0)
        clipped_probs = np.clip(probabilities, epsilon, 1 - epsilon)
        log_loss = -np.mean(np.sum(one_hot_labels * np.log(clipped_probs), axis=1))
        calibration_metrics['log_loss'] = float(log_loss)
        
        # Maximum predicted probability statistics
        max_probs = probabilities.max(axis=1)
        calibration_metrics['confidence_stats'] = {
            'mean_max_prob': float(max_probs.mean()),
            'std_max_prob': float(max_probs.std()),
            'min_max_prob': float(max_probs.min()),
            'max_max_prob': float(max_probs.max())
        }
        
        return calibration_metrics
    
    def _print_evaluation_summary(self, results: Dict):
        """Print evaluation summary."""
        tag = results['tag']
        accuracy = results['accuracy']
        
        print(f"\n📊 {tag} Evaluation Results:")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Random baseline
        random_accuracy = 1.0 / self.config.n_quantiles
        improvement = (accuracy - random_accuracy) / random_accuracy * 100
        print(f"   Random Baseline: {random_accuracy:.3f} ({random_accuracy*100:.1f}%)")
        print(f"   Improvement: {improvement:+.1f}%")
        
        # Confusion matrix summary
        cm = results['confusion_matrix']
        print(f"   Confusion Matrix:")
        print(f"     Diagonal sum: {cm.diagonal().sum()}/{cm.sum()}")
        
        # Prediction distribution
        pred_dist = results['prediction_distribution']
        print(f"   Prediction Distribution:")
        for key, value in pred_dist.items():
            print(f"     {key}: {value['count']} ({value['percentage']:.1f}%)")
        
        # Calibration
        if 'overall_ece' in results['reliability_stats']:
            ece = results['reliability_stats']['overall_ece']
            print(f"   Expected Calibration Error: {ece:.3f}")
        
        brier = results['calibration_metrics']['brier_score']
        log_loss = results['calibration_metrics']['log_loss']
        print(f"   Brier Score: {brier:.3f}")
        print(f"   Log Loss: {log_loss:.3f}")
    
    def _generate_evaluation_plots(self, probabilities: np.ndarray, actual_labels: np.ndarray, 
                                  predictions: np.ndarray, tag: str):
        """Generate evaluation plots."""
        try:
            # Check if we're in a subprocess/thread - if so, skip plotting
            import threading
            if threading.current_thread() != threading.main_thread():
                print(f"⚠️  Skipping plot generation in subprocess for {tag}")
                return
            
            # Use non-interactive backend for headless operation
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{tag} Set Evaluation', fontsize=16)
            
            # 1. Confusion Matrix
            cm = confusion_matrix(actual_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # 2. Reliability Diagram
            self._plot_reliability_diagram(probabilities, actual_labels, axes[0, 1])
            
            # 3. Prediction Distribution
            self._plot_prediction_distribution(predictions, axes[1, 0])
            
            # 4. Confidence Distribution
            max_probs = probabilities.max(axis=1)
            axes[1, 1].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Prediction Confidence Distribution')
            axes[1, 1].set_xlabel('Maximum Probability')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(max_probs.mean(), color='red', linestyle='--', 
                              label=f'Mean: {max_probs.mean():.3f}')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.config.cache_dir / f"evaluation_{tag.lower()}_{self.config.symbol}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Evaluation plot saved: {plot_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate evaluation plots: {e}")
    
    def _plot_reliability_diagram(self, probabilities: np.ndarray, actual_labels: np.ndarray, ax):
        """Plot reliability diagram for the most confident class."""
        try:
            # Use the class with most predictions for reliability diagram
            pred_class = probabilities.argmax(axis=1)
            most_common_class = np.bincount(pred_class).argmax()
            
            binary_labels = (actual_labels == most_common_class).astype(int)
            class_probs = probabilities[:, most_common_class]
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_labels, class_probs, n_bins=10
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-', label=f'Quantile {most_common_class}')
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Reliability Diagram')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reliability Diagram (Error)')
    
    def _plot_prediction_distribution(self, predictions: np.ndarray, ax):
        """Plot distribution of predictions."""
        unique, counts = np.unique(predictions, return_counts=True)
        
        # Ensure all quantiles are represented
        all_quantiles = np.arange(self.config.n_quantiles)
        all_counts = np.zeros(self.config.n_quantiles)
        
        for i, q in enumerate(all_quantiles):
            if q in unique:
                all_counts[i] = counts[unique == q][0]
        
        bars = ax.bar(all_quantiles, all_counts, alpha=0.7, edgecolor='black')
        ax.set_title('Prediction Distribution')
        ax.set_xlabel('Predicted Quantile')
        ax.set_ylabel('Count')
        ax.set_xticks(all_quantiles)
        
        # Add percentage labels on bars
        total = all_counts.sum()
        for bar, count in zip(bars, all_counts):
            height = bar.get_height()
            if height > 0:
                percentage = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{percentage:.1f}%',
                       ha='center', va='bottom')


def evaluate_model(model: nn.Module, dataset, config: MLConfig, tag: str = "Test") -> Dict:
    """
    Convenience function for model evaluation.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        config: ML configuration
        tag: Label for evaluation
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate(model, dataset, tag)


def compare_models(models: Dict[str, nn.Module], dataset, config: MLConfig) -> Dict:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of {name: model} pairs
        dataset: Dataset to evaluate on
        config: ML configuration
        
    Returns:
        Comparison results
    """
    print("🔀 Comparing models...")
    
    results = {}
    for name, model in models.items():
        print(f"\n📊 Evaluating {name}...")
        evaluator = ModelEvaluator(config)
        results[name] = evaluator.evaluate(model, dataset, name)
    
    # Create comparison summary
    comparison = {
        'model_names': list(models.keys()),
        'accuracies': [results[name]['accuracy'] for name in models.keys()],
        'brier_scores': [results[name]['calibration_metrics']['brier_score'] for name in models.keys()],
        'log_losses': [results[name]['calibration_metrics']['log_loss'] for name in models.keys()],
        'individual_results': results
    }
    
    # Print comparison
    print("\n🏆 Model Comparison Summary:")
    print("=" * 50)
    for i, name in enumerate(models.keys()):
        acc = comparison['accuracies'][i]
        brier = comparison['brier_scores'][i]
        log_loss = comparison['log_losses'][i]
        print(f"{name:15s} | Acc: {acc:.3f} | Brier: {brier:.3f} | LogLoss: {log_loss:.3f}")
    
    # Find best model
    best_acc_idx = np.argmax(comparison['accuracies'])
    best_model = comparison['model_names'][best_acc_idx]
    print(f"\n🥇 Best model (by accuracy): {best_model}")
    
    return comparison 