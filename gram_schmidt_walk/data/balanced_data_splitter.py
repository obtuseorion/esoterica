"""
Balanced Dataset Splitter

This module provides functionality to create balanced training and validation
splits using the Gram-Schmidt Walk algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats

from ..core.gram_schmidt_walk import GramSchmidtWalk
from .data_processor import DataProcessor


class BalancedDataSplitter:
    """
    Creates balanced dataset splits using the Gram-Schmidt Walk algorithm.
    
    This ensures that training and validation sets both represent 
    the original data distribution well.
    """
    
    def __init__(self, 
                 validation_size=0.2,
                 random_state=None,
                 balance_features=True,
                 balance_labels=True,
                 balance_statistics=True,
                 processor_kwargs=None):
        """
        Initialize the dataset splitter.
        
        Args:
            validation_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            balance_features: Whether to balance feature distributions
            balance_labels: Whether to balance label distributions
            balance_statistics: Whether to balance statistical properties
            processor_kwargs: Additional arguments for DataProcessor
        """
        self.validation_size = validation_size
        self.random_state = random_state
        self.balance_features = balance_features
        self.balance_labels = balance_labels
        self.balance_statistics = balance_statistics
        
        processor_kwargs = processor_kwargs or {}
        processor_kwargs.update({
            'balance_features': balance_features,
            'balance_labels': balance_labels,
            'balance_statistics': balance_statistics
        })
        self.processor = DataProcessor(**processor_kwargs)
        
        self.result = None
        
    def split(self, X, y=None, verbose=False):
        """
        Create balanced training and validation splits.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            verbose: Whether to print progress information
            
        Returns:
            X_train, X_val, y_train, y_val: Balanced dataset splits
        """
        n_samples = len(X)
        val_count = int(n_samples * self.validation_size)
        
        if verbose:
            print(f"Creating balanced split with {n_samples - val_count} training and {val_count} validation samples")
        
        balance_vectors = self.processor.create_balance_vectors(X, y)
        
        if verbose:
            print(f"Created {balance_vectors.shape[1]} balance vectors")
            
        initial_coloring = np.zeros(n_samples)
        
        if verbose:
            print("Running Gram-Schmidt Walk...")
            
        gsw = GramSchmidtWalk(balance_vectors, initial_coloring=initial_coloring)
        self.result = gsw.run(verbose=verbose)
        final_coloring = self.result["coloring"]
        
        train_mask = final_coloring > 0
        val_mask = ~train_mask
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        if len(val_indices) != val_count:
            if len(val_indices) < val_count:
                np.random.seed(self.random_state)
                move_to_val = np.random.choice(
                    train_indices,
                    size=val_count - len(val_indices),
                    replace=False
                )
                val_indices = np.append(val_indices, move_to_val)
                train_indices = np.setdiff1d(train_indices, move_to_val)
            else:
                np.random.seed(self.random_state)
                move_to_train = np.random.choice(
                    val_indices,
                    size=len(val_indices) - val_count,
                    replace=False
                )
                train_indices = np.append(train_indices, move_to_train)
                val_indices = np.setdiff1d(val_indices, move_to_train)
        
        X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
        X_val = X.iloc[val_indices] if hasattr(X, 'iloc') else X[val_indices]
        
        if y is not None:
            y_train = y[train_indices] if isinstance(y, np.ndarray) else y.iloc[train_indices]
            y_val = y[val_indices] if isinstance(y, np.ndarray) else y.iloc[val_indices]
            return X_train, X_val, y_train, y_val
        else:
            return X_train, X_val
            
    def split_indices(self, X, y=None, verbose=False):
        """
        Create balanced training and validation splits and return the indices.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            verbose: Whether to print progress information
            
        Returns:
            train_indices, val_indices: Indices for training and validation sets
        """
        n_samples = len(X)
        val_count = int(n_samples * self.validation_size)
        
        if verbose:
            print(f"Creating balanced split with {n_samples - val_count} training and {val_count} validation samples")
        
        balance_vectors = self.processor.create_balance_vectors(X, y)
        
        if verbose:
            print(f"Created {balance_vectors.shape[1]} balance vectors")
            
        initial_coloring = np.zeros(n_samples)
        
        if verbose:
            print("Running Gram-Schmidt Walk...")
            
        gsw = GramSchmidtWalk(balance_vectors, initial_coloring=initial_coloring)
        self.result = gsw.run(verbose=verbose)
        final_coloring = self.result["coloring"]
        
        train_mask = final_coloring > 0
        val_mask = ~train_mask
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        if len(val_indices) != val_count:
            if len(val_indices) < val_count:
                np.random.seed(self.random_state)
                move_to_val = np.random.choice(
                    train_indices,
                    size=val_count - len(val_indices),
                    replace=False
                )
                val_indices = np.append(val_indices, move_to_val)
                train_indices = np.setdiff1d(train_indices, move_to_val)
            else:
                np.random.seed(self.random_state)
                move_to_train = np.random.choice(
                    val_indices,
                    size=len(val_indices) - val_count,
                    replace=False
                )
                train_indices = np.append(train_indices, move_to_train)
                val_indices = np.setdiff1d(val_indices, move_to_train)
        
        return train_indices, val_indices
    
    def evaluate_balance(self, X_orig, X_train, X_val, y_orig=None, y_train=None, y_val=None):
        """
        Evaluate how balanced the splits are.
        
        Args:
            X_orig: Original dataset
            X_train: Training split
            X_val: Validation split
            y_orig, y_train, y_val: Optional target variables
            
        Returns:
            Dictionary with balance metrics
        """
        metrics = {}
        
        X_orig_proc = self.processor.transform(X_orig)
        X_train_proc = self.processor.transform(X_train)
        X_val_proc = self.processor.transform(X_val)
        
        if self.balance_features:
            orig_means = np.mean(X_orig_proc, axis=0)
            train_means = np.mean(X_train_proc, axis=0)
            val_means = np.mean(X_val_proc, axis=0)
            
            metrics['mean_diff_train'] = np.linalg.norm(orig_means - train_means)
            metrics['mean_diff_val'] = np.linalg.norm(orig_means - val_means)
            
            orig_std = np.std(X_orig_proc, axis=0)
            train_std = np.std(X_train_proc, axis=0)
            val_std = np.std(X_val_proc, axis=0)
            
            metrics['std_diff_train'] = np.linalg.norm(orig_std - train_std)
            metrics['std_diff_val'] = np.linalg.norm(orig_std - val_std)
            
            if X_orig_proc.shape[1] < 50:
                orig_cov = np.cov(X_orig_proc, rowvar=False)
                train_cov = np.cov(X_train_proc, rowvar=False)
                val_cov = np.cov(X_val_proc, rowvar=False)
                
                metrics['cov_diff_train'] = np.linalg.norm(orig_cov - train_cov, 'fro') / np.linalg.norm(orig_cov, 'fro')
                metrics['cov_diff_val'] = np.linalg.norm(orig_cov - val_cov, 'fro') / np.linalg.norm(orig_cov, 'fro')
        
        if y_orig is not None and self.balance_labels:
            if isinstance(y_orig, np.ndarray) and y_orig.ndim == 1:
                y_orig, y_train, y_val = np.array(y_orig), np.array(y_train), np.array(y_val)
                
                classes = np.unique(y_orig)
                orig_dist = np.array([np.mean(y_orig == c) for c in classes])
                train_dist = np.array([np.mean(y_train == c) for c in classes])
                val_dist = np.array([np.mean(y_val == c) for c in classes])
                
                metrics['label_diff_train'] = np.linalg.norm(orig_dist - train_dist, 1)
                metrics['label_diff_val'] = np.linalg.norm(orig_dist - val_dist, 1)
            elif isinstance(y_orig, np.ndarray) and y_orig.ndim == 2:
                y_orig, y_train, y_val = np.array(y_orig), np.array(y_train), np.array(y_val)
                
                orig_means = np.mean(y_orig, axis=0)
                train_means = np.mean(y_train, axis=0)
                val_means = np.mean(y_val, axis=0)
                
                metrics['label_mean_diff_train'] = np.linalg.norm(orig_means - train_means)
                metrics['label_mean_diff_val'] = np.linalg.norm(orig_means - val_means)
        
        num_trials = 10
        random_metrics = {'mean_diff_val': [], 'std_diff_val': []}
        
        for i in range(num_trials):
            _, X_val_random, _, y_val_random = train_test_split(
                X_orig, y_orig, test_size=self.validation_size, random_state=i
            )
            
            X_val_random_proc = self.processor.transform(X_val_random)
            
            val_means_random = np.mean(X_val_random_proc, axis=0)
            random_metrics['mean_diff_val'].append(np.linalg.norm(orig_means - val_means_random))
            
            val_std_random = np.std(X_val_random_proc, axis=0)
            random_metrics['std_diff_val'].append(np.linalg.norm(orig_std - val_std_random))
        
        metrics['mean_diff_improvement'] = np.mean(random_metrics['mean_diff_val']) / metrics['mean_diff_val']
        metrics['std_diff_improvement'] = np.mean(random_metrics['std_diff_val']) / metrics['std_diff_val']
        
        metrics['overall_balance_score'] = (metrics['mean_diff_improvement'] + metrics['std_diff_improvement']) / 2
        
        return metrics
    
    def compare_with_random(self, X, y=None, num_trials=10):
        """
        Compare balanced splitting with random splitting.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            num_trials: Number of random splits to compare against
            
        Returns:
            Dictionary with comparison metrics
        """
        if y is not None:
            X_train_bal, X_val_bal, y_train_bal, y_val_bal = self.split(X, y)
        else:
            X_train_bal, X_val_bal = self.split(X)
            y_train_bal, y_val_bal = None, None
        
        balanced_metrics = self.evaluate_balance(X, X_train_bal, X_val_bal, y, y_train_bal, y_val_bal)
        
        random_metrics = []
        for i in range(num_trials):
            if y is not None:
                X_train_rand, X_val_rand, y_train_rand, y_val_rand = train_test_split(
                    X, y, test_size=self.validation_size, random_state=i
                )
            else:
                X_train_rand, X_val_rand = train_test_split(
                    X, test_size=self.validation_size, random_state=i
                )
                y_train_rand, y_val_rand = None, None
            
            metrics = self.evaluate_balance(X, X_train_rand, X_val_rand, y, y_train_rand, y_val_rand)
            random_metrics.append(metrics)
        
        avg_random_metrics = {}
        for key in balanced_metrics.keys():
            values = [m.get(key, 0) for m in random_metrics]
            avg_random_metrics[key] = np.mean(values)
        
        improvement = {}
        for key in balanced_metrics.keys():
            if key in avg_random_metrics and avg_random_metrics[key] > 0:
                if 'diff' in key:
                    improvement[key] = avg_random_metrics[key] / balanced_metrics[key]
                else:
                    improvement[key] = balanced_metrics[key] / avg_random_metrics[key]
        
        comparison = {
            'balanced': balanced_metrics,
            'random_avg': avg_random_metrics,
            'improvement': improvement
        }
        
        return comparison
    
    def visualize_comparison(self, comparison, title="Balanced vs Random Splitting"):
        """
        Visualize comparison between balanced and random splitting.
        
        Args:
            comparison: Output from compare_with_random
            title: Plot title
        """
        metrics = ['mean_diff_val', 'std_diff_val', 'cov_diff_val', 'label_diff_val']
        available_metrics = [m for m in metrics if m in comparison['balanced']]
        
        labels = [m.replace('_diff_val', '').replace('_', ' ').title() for m in available_metrics]
        balanced_values = [comparison['balanced'][m] for m in available_metrics]
        random_values = [comparison['random_avg'][m] for m in available_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, balanced_values, width, label='Balanced Split')
        ax.bar(x + width/2, random_values, width, label='Random Split')
        
        ax.set_ylabel('Difference (lower is better)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        for i, metric in enumerate(available_metrics):
            if metric in comparison['improvement']:
                improvement = comparison['improvement'][metric]
                ax.annotate(f'{improvement:.2f}x better', 
                           xy=(i, balanced_values[i]), 
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center')
        
        plt.tight_layout()
        plt.savefig('balance_comparison.png')
        plt.close()
        
        print(f"Comparison plot saved as 'balance_comparison.png'")
    
    def evaluate_indices(self, X, train_indices, val_indices, y=None):
        """
        Evaluate balance metrics for splits defined by indices.
        
        Args:
            X: Feature matrix
            train_indices: Indices for training set
            val_indices: Indices for validation set
            y: Optional target vector
            
        Returns:
            Dictionary with balance metrics
        """
        X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
        X_val = X.iloc[val_indices] if hasattr(X, 'iloc') else X[val_indices]
        
        if y is not None:
            y_train = y[train_indices] if isinstance(y, np.ndarray) else y.iloc[train_indices]
            y_val = y[val_indices] if isinstance(y, np.ndarray) else y.iloc[val_indices]
            return self.evaluate_balance(X, X_train, X_val, y, y_train, y_val)
        else:
            return self.evaluate_balance(X, X_train, X_val)


def quick_balanced_split(X, y=None, validation_size=0.2, random_state=None, verbose=False):
    """
    Quick helper function to create balanced splits.
    
    Args:
        X: Feature matrix
        y: Optional target vector
        validation_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        X_train, X_val, y_train, y_val: Balanced dataset splits
    """
    splitter = BalancedDataSplitter(
        validation_size=validation_size,
        random_state=random_state
    )
    
    if y is not None:
        return splitter.split(X, y, verbose=verbose)
    else:
        return splitter.split(X, verbose=verbose)


def quick_balanced_split_indices(X, y=None, validation_size=0.2, random_state=None, verbose=False):
    """
    Quick helper function to create balanced split indices.
    
    Args:
        X: Feature matrix
        y: Optional target vector
        validation_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        train_indices, val_indices: Indices for balanced dataset splits
    """
    splitter = BalancedDataSplitter(
        validation_size=validation_size,
        random_state=random_state
    )
    
    return splitter.split_indices(X, y, verbose=verbose)

#testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, 
        n_features=20,
        n_informative=2, 
        n_redundant=10,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    splitter = BalancedDataSplitter(validation_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val = splitter.split(X, y, verbose=True)
    
    comparison = splitter.compare_with_random(X, y)
    
    print("\nComparison between balanced and random splitting:")
    for metric in ['mean_diff_val', 'std_diff_val']:
        balanced = comparison['balanced'][metric]
        random = comparison['random_avg'][metric]
        improvement = comparison['improvement'][metric]
        print(f"{metric}: Balanced={balanced:.4f}, Random={random:.4f}, {improvement:.2f}x better")
    
    splitter.visualize_comparison(comparison)
    
    print("\nOriginal class distribution:")
    print(f"Class 0: {np.mean(y == 0):.4f}")
    print(f"Class 1: {np.mean(y == 1):.4f}")
    
    print("\nTraining set class distribution:")
    print(f"Class 0: {np.mean(y_train == 0):.4f}")
    print(f"Class 1: {np.mean(y_train == 1):.4f}")
    
    print("\nValidation set class distribution:")
    print(f"Class 0: {np.mean(y_val == 0):.4f}")
    print(f"Class 1: {np.mean(y_val == 1):.4f}")
    
    print("\nTesting split_indices functionality:")
    train_indices, val_indices = splitter.split_indices(X, y, verbose=True)
    
    print(f"Number of training indices: {len(train_indices)}")
    print(f"Number of validation indices: {len(val_indices)}")
    
    X_train_indices = X[train_indices]
    X_val_indices = X[val_indices]
    y_train_indices = y[train_indices]
    y_val_indices = y[val_indices]
    
    print("\nClass distribution from indices:")
    print(f"Class 0 (train): {np.mean(y_train_indices == 0):.4f}")
    print(f"Class 1 (train): {np.mean(y_train_indices == 1):.4f}")
    print(f"Class 0 (val): {np.mean(y_val_indices == 0):.4f}")
    print(f"Class 1 (val): {np.mean(y_val_indices == 1):.4f}")