#!/usr/bin/env python3
"""
Basic Usage Examples for Gram-Schmidt Walk Algorithm

This script demonstrates the basic functionality of the Gram-Schmidt Walk
implementation with simple, clear examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris

# Import from the package
from gram_schmidt_walk import (
    GramSchmidtWalk, 
    BalancedDataSplitter, 
    DataProcessor,
    quick_balanced_split
)
from gram_schmidt_walk.examples import ToyDatasetGenerator


def example_1_basic_vector_balancing():
    """
    Example 1: Basic vector balancing with random vectors.
    """
    print("="*60)
    print("EXAMPLE 1: Basic Vector Balancing")
    print("="*60)
    
    # Create random vectors with unit norm
    np.random.seed(42)
    n_vectors, n_dims = 20, 8
    vectors = np.random.randn(n_vectors, n_dims)
    
    # Normalize to have L2 norm ≤ 1
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors / np.maximum(norms[:, np.newaxis], 1.0)
    
    print(f"Created {n_vectors} vectors in {n_dims} dimensions")
    print(f"Vector norms: min={np.min(norms):.3f}, max={np.max(norms):.3f}")
    
    # Run Gram-Schmidt Walk
    gsw = GramSchmidtWalk(vectors)
    result = gsw.run(verbose=True)
    
    # Display results
    print(f"\nResults:")
    print(f"  Final coloring: {result['coloring']}")
    print(f"  L-infinity discrepancy: {result['discrepancy']['l_inf']:.4f}")
    print(f"  L2 discrepancy: {result['discrepancy']['l2']:.4f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Runtime: {result['time']:.4f} seconds")
    
    # Verify the coloring
    discrepancy_vector = vectors.T @ result['coloring']
    print(f"  Verification - max |component|: {np.max(np.abs(discrepancy_vector)):.4f}")


def example_2_balanced_dataset_splitting():
    """
    Example 2: Balanced dataset splitting for machine learning.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Balanced Dataset Splitting")
    print("="*60)
    
    # Create an imbalanced dataset
    X, y = make_classification(
        n_samples=200, 
        n_features=10,
        n_informative=5,
        n_classes=2,
        weights=[0.8, 0.2],  # Imbalanced classes
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)} (ratio: {np.mean(y):.3f})")
    
    # Create balanced splits
    X_train, X_val, y_train, y_val = quick_balanced_split(
        X, y, validation_size=0.25, random_state=42, verbose=True
    )
    
    print(f"\nSplit Results:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Train class ratio: {np.mean(y_train):.3f}")
    print(f"  Validation class ratio: {np.mean(y_val):.3f}")
    
    # Compare with random splitting
    from sklearn.model_selection import train_test_split
    X_train_rand, X_val_rand, y_train_rand, y_val_rand = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"\nComparison with Random Split:")
    print(f"  Random train class ratio: {np.mean(y_train_rand):.3f}")
    print(f"  Random val class ratio: {np.mean(y_val_rand):.3f}")
    print(f"  Balanced split preserves class distribution better!")


def example_3_data_preprocessing():
    """
    Example 3: Data preprocessing pipeline for mixed data types.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Preprocessing Pipeline")
    print("="*60)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)} (counts: {np.bincount(y)})")
    
    # Create data processor
    processor = DataProcessor(
        scaling='robust',
        balance_features=True,
        balance_labels=True,
        balance_statistics=True
    )
    
    # Create balance vectors
    balance_vectors = processor.create_balance_vectors(X, y)
    
    print(f"\nPreprocessing Results:")
    print(f"  Balance vectors shape: {balance_vectors.shape}")
    print(f"  Max vector norm: {np.max(np.linalg.norm(balance_vectors, axis=1)):.3f}")
    print(f"  Min vector norm: {np.min(np.linalg.norm(balance_vectors, axis=1)):.3f}")
    
    # Use balance vectors with Gram-Schmidt Walk
    gsw = GramSchmidtWalk(balance_vectors)
    result = gsw.run(verbose=False)
    
    print(f"  GSW discrepancy: {result['discrepancy']['l_inf']:.4f}")
    print(f"  GSW iterations: {result['iterations']}")


def example_4_toy_datasets():
    """
    Example 4: Testing with various toy datasets.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Toy Dataset Comparison")
    print("="*60)
    
    generator = ToyDatasetGenerator(random_state=42)
    
    # Test different dataset types
    datasets = {
        'Orthogonal': generator.create_orthogonal_vectors(n=15, m=8),
        'Clustered': generator.create_clustered_vectors(n=20, m=10, n_clusters=3)[0],
        'Sparse': generator.create_sparse_vectors(n=18, m=15, sparsity=0.2),
        'Adversarial': generator.create_adversarial_vectors(n=12, m=8)
    }
    
    results = {}
    
    for name, vectors in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"  Shape: {vectors.shape}")
        
        gsw = GramSchmidtWalk(vectors)
        result = gsw.run(verbose=False)
        
        discrepancy = result['discrepancy']['l_inf']
        n = vectors.shape[0]
        theoretical_bound = np.sqrt(np.log(n))
        
        print(f"  L-inf discrepancy: {discrepancy:.4f}")
        print(f"  √log(n) bound: {theoretical_bound:.4f}")
        print(f"  Ratio to bound: {discrepancy/theoretical_bound:.2f}")
        print(f"  Runtime: {result['time']:.4f}s")
        
        results[name] = {
            'discrepancy': discrepancy,
            'bound': theoretical_bound,
            'ratio': discrepancy/theoretical_bound,
            'time': result['time']
        }
    
    # Find best and worst performers
    best = min(results.items(), key=lambda x: x[1]['discrepancy'])
    worst = max(results.items(), key=lambda x: x[1]['discrepancy'])
    
    print(f"\nSummary:")
    print(f"  Best performance: {best[0]} (discrepancy: {best[1]['discrepancy']:.4f})")
    print(f"  Worst performance: {worst[0]} (discrepancy: {worst[1]['discrepancy']:.4f})")


def example_5_visualization():
    """
    Example 5: Create visualizations of algorithm performance.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Algorithm Visualization")
    print("="*60)
    
    # Create test vectors
    np.random.seed(42)
    vectors = np.random.randn(30, 10)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors / np.maximum(norms[:, np.newaxis], 1.0)
    
    # Run algorithm
    gsw = GramSchmidtWalk(vectors)
    result = gsw.run(verbose=False)
    
    # Plot discrepancy evolution
    history = result['discrepancy_history']
    iterations = list(range(1, len(history) + 1))
    l_inf_values = [d['l_inf'] for d in history]
    l2_values = [d['l2'] for d in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, l_inf_values, 'b-', label='L-∞ discrepancy', linewidth=2)
    plt.plot(iterations, l2_values, 'r--', label='L2 discrepancy', linewidth=2)
    
    # Add theoretical bounds
    n = vectors.shape[0]
    plt.axhline(y=np.sqrt(40), color='g', linestyle=':', 
                label='Paper bound (√40 ≈ 6.32)', alpha=0.7)
    plt.axhline(y=np.sqrt(np.log(n)), color='orange', linestyle=':', 
                label=f'√log(n) ≈ {np.sqrt(np.log(n)):.2f}', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Discrepancy')
    plt.title('Gram-Schmidt Walk: Discrepancy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('basic_usage_discrepancy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved as 'basic_usage_discrepancy.png'")
    print(f"Final L-∞ discrepancy: {result['discrepancy']['l_inf']:.4f}")
    print(f"Algorithm completed in {result['iterations']} iterations")


def main():
    """
    Run all examples.
    """
    print("GRAM-SCHMIDT WALK: Basic Usage Examples")
    print("=" * 80)
    
    try:
        example_1_basic_vector_balancing()
        example_2_balanced_dataset_splitting()
        example_3_data_preprocessing()
        example_4_toy_datasets()
        example_5_visualization()
        
        print("\n" + "=" * 80)
        print("🎉 All examples completed successfully!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install the package with: pip install -e .")
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()