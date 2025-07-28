"""
Toy Dataset Generator for Testing Gram-Schmidt Walk Algorithm

This module provides various synthetic datasets designed to test different
aspects of the Gram-Schmidt Walk algorithm, from basic functionality to
challenging edge cases.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ToyDatasetGenerator:
    """
    Generate synthetic datasets to test various aspects of the Gram-Schmidt Walk algorithm.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the dataset generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_orthogonal_vectors(self, n=20, m=10):
        """
        Create a set of orthogonal vectors - ideal case for the algorithm.
        
        Args:
            n: Number of vectors
            m: Dimension of each vector
            
        Returns:
            Array of normalized orthogonal vectors
        """
        A = np.random.randn(n, m)
        
        Q = np.zeros_like(A)
        for i in range(min(n, m)):
            v = A[i].copy()
            for j in range(i):
                v = v - np.dot(v, Q[j]) * Q[j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                Q[i] = v / norm
        
        for i in range(min(n, m), n):
            v = np.random.randn(m)
            Q[i] = v / np.linalg.norm(v)
        
        return Q
    
    def create_clustered_vectors(self, n=100, m=20, n_clusters=3, cluster_std=0.3):
        """
        Create vectors that form distinct clusters - tests algorithm on structured data.
        
        Args:
            n: Number of vectors
            m: Dimension of each vector
            n_clusters: Number of clusters
            cluster_std: Standard deviation within clusters
            
        Returns:
            Tuple of (vectors, cluster_labels)
        """
        centers = np.random.randn(n_clusters, m)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        
        cluster_labels = np.random.choice(n_clusters, size=n)
        
        vectors = np.zeros((n, m))
        for i in range(n):
            cluster_id = cluster_labels[i]
            noise = np.random.randn(m) * cluster_std
            vectors[i] = centers[cluster_id] + noise
            vectors[i] = vectors[i] / max(np.linalg.norm(vectors[i]), 1.0)
        
        return vectors, cluster_labels
    
    def create_sparse_vectors(self, n=50, m=100, sparsity=0.1):
        """
        Create sparse vectors - most entries are zero.
        
        Args:
            n: Number of vectors
            m: Dimension of each vector
            sparsity: Fraction of non-zero entries
            
        Returns:
            Array of sparse normalized vectors
        """
        vectors = np.zeros((n, m))
        
        for i in range(n):
            n_nonzero = max(1, int(m * sparsity))
            positions = np.random.choice(m, size=n_nonzero, replace=False)
            
            vectors[i, positions] = np.random.randn(n_nonzero)
            
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] = vectors[i] / norm
        
        return vectors
    
    def create_adversarial_vectors(self, n=30, m=15):
        """
        Create vectors designed to be challenging for discrepancy minimization.
        Based on constructions that achieve worst-case bounds.
        
        Args:
            n: Number of vectors
            m: Dimension of each vector
            
        Returns:
            Array of adversarial vectors
        """
        vectors = np.zeros((n, m))
        
        for i in range(min(n, m)):
            vectors[i, i] = 1.0
        
        for i in range(min(n, m), n):
            coeffs = np.random.randn(min(i, m))
            coeffs = coeffs / np.linalg.norm(coeffs)
            
            for j in range(min(i, m)):
                vectors[i] += coeffs[j] * vectors[j]
            
            vectors[i] += 0.1 * np.random.randn(m)
            
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] = vectors[i] / min(norm, 1.0)
        
        return vectors
    
    def create_imbalanced_dataset(self, n=200, n_features=10, imbalance_ratio=0.1):
        """
        Create an imbalanced classification dataset to test balanced splitting.
        
        Args:
            n: Number of samples
            n_features: Number of features
            imbalance_ratio: Ratio of minority to majority class
            
        Returns:
            Tuple of (X, y) with imbalanced classes
        """
        n_minority = int(n * imbalance_ratio)
        n_majority = n - n_minority
        
        X_majority = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_majority
        )
        y_majority = np.zeros(n_majority)
        
        mean_shift = np.ones(n_features) * 2
        X_minority = np.random.multivariate_normal(
            mean=mean_shift,
            cov=0.5 * np.eye(n_features),
            size=n_minority
        )
        y_minority = np.ones(n_minority)
        
        X = np.vstack([X_majority, X_minority])
        y = np.hstack([y_majority, y_minority])
        
        indices = np.random.permutation(n)
        X, y = X[indices], y[indices]
        
        return X, y
    
    def create_multimodal_dataset(self, n=150, n_features=8, n_modes=4):
        """
        Create dataset with multiple modes/peaks in feature distribution.
        
        Args:
            n: Number of samples
            n_features: Number of features
            n_modes: Number of modes in the distribution
            
        Returns:
            Tuple of (X, y, mode_labels)
        """
        X, mode_labels = make_blobs(
            n_samples=n,
            centers=n_modes,
            n_features=n_features,
            cluster_std=1.5,
            center_box=(-3.0, 3.0),
            random_state=self.random_state
        )
        
        y = np.zeros(n)
        for i in range(n_modes):
            mask = mode_labels == i

            if i < n_modes // 2:
                y[mask] = np.random.choice([0, 1], size=np.sum(mask), p=[0.8, 0.2])
            else:
                y[mask] = np.random.choice([0, 1], size=np.sum(mask), p=[0.3, 0.7])
        
        return X, y, mode_labels
    
    def create_correlated_features_dataset(self, n=100, n_features=20, correlation=0.8):
        """
        Create dataset with highly correlated features to test preprocessing.
        
        Args:
            n: Number of samples
            n_features: Number of features
            correlation: Correlation coefficient between features
            
        Returns:
            Tuple of (X, y) with correlated features
        """
        corr_matrix = np.full((n_features, n_features), correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        X = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=corr_matrix,
            size=n
        )
        
        y = np.sign(X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n) * 0.1)
        y = (y + 1) / 2  # Convert to 0/1
        
        return X, y
    
    def create_mixed_types_dataset(self, n=200):
        """
        Create dataset with mixed data types (numerical, categorical, missing values).
        
        Args:
            n: Number of samples
            
        Returns:
            DataFrame with mixed types and some missing values
        """
        data = {}
        
        data['numeric1'] = np.random.randn(n)
        data['numeric2'] = np.random.exponential(2, n)
        data['numeric3'] = np.random.uniform(-5, 5, n)
        
        data['category1'] = np.random.choice(['A', 'B', 'C', 'D'], size=n, p=[0.4, 0.3, 0.2, 0.1])
        data['category2'] = np.random.choice(['X', 'Y', 'Z'], size=n)
        data['binary'] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
        
        target_score = (
            0.5 * data['numeric1'] + 
            0.3 * data['numeric2'] + 
            (data['category1'] == 'A').astype(float) * 0.4 +
            data['binary'] * 0.3 +
            np.random.randn(n) * 0.2
        )
        data['target'] = (target_score > np.median(target_score)).astype(int)
        
        df = pd.DataFrame(data)
        
        missing_mask = np.random.random((n, len(df.columns))) < 0.05  
        for i, col in enumerate(df.columns):
            if col != 'target':  
                df.loc[missing_mask[:, i], col] = np.nan if df[col].dtype == 'object' else np.nan
        
        return df


def create_test_suite():
    """
    Create a comprehensive test suite with various challenging datasets.
    
    Returns:
        Dictionary of test datasets with metadata
    """
    generator = ToyDatasetGenerator(random_state=42)
    
    test_suite = {}
    
    # 1. Orthogonal vectors (ideal case)
    test_suite['orthogonal'] = {
        'vectors': generator.create_orthogonal_vectors(n=30, m=15),
        'description': 'Orthogonal vectors - should achieve near-zero discrepancy',
        'expected_performance': 'Excellent (discrepancy < 0.5)'
    }
    
    # 2. Clustered vectors
    vectors, labels = generator.create_clustered_vectors(n=60, m=20, n_clusters=3)
    test_suite['clustered'] = {
        'vectors': vectors,
        'labels': labels,
        'description': 'Clustered vectors - tests structured data handling',
        'expected_performance': 'Good (discrepancy < 2.0)'
    }
    
    # 3. Sparse vectors
    test_suite['sparse'] = {
        'vectors': generator.create_sparse_vectors(n=40, m=80, sparsity=0.1),
        'description': 'Sparse vectors - tests high-dimensional sparse data',
        'expected_performance': 'Moderate (discrepancy < 3.0)'
    }
    
    # 4. Adversarial vectors
    test_suite['adversarial'] = {
        'vectors': generator.create_adversarial_vectors(n=25, m=12),
        'description': 'Adversarial vectors - designed to be challenging',
        'expected_performance': 'Poor (discrepancy may exceed theoretical bounds)'
    }
    
    # 5. Imbalanced dataset for splitting
    X, y = generator.create_imbalanced_dataset(n=150, imbalance_ratio=0.15)
    test_suite['imbalanced_split'] = {
        'X': X,
        'y': y,
        'description': 'Imbalanced dataset for testing balanced splitting',
        'expected_performance': 'Should preserve class ratios in splits'
    }
    
    # 6. Multimodal dataset
    X, y, modes = generator.create_multimodal_dataset(n=120, n_modes=4)
    test_suite['multimodal'] = {
        'X': X,
        'y': y,
        'modes': modes,
        'description': 'Multimodal distribution - tests complex data structure',
        'expected_performance': 'Should balance across all modes'
    }
    
    # 7. Correlated features
    X, y = generator.create_correlated_features_dataset(n=80, correlation=0.85)
    test_suite['correlated'] = {
        'X': X,
        'y': y,
        'description': 'Highly correlated features - tests preprocessing',
        'expected_performance': 'Requires dimensionality reduction'
    }
    
    # 8. Mixed types dataset
    test_suite['mixed_types'] = {
        'data': generator.create_mixed_types_dataset(n=100),
        'description': 'Mixed data types with missing values',
        'expected_performance': 'Tests complete preprocessing pipeline'
    }
    
    return test_suite


def run_test_suite():
    """
    Run the complete test suite and generate comprehensive reports.
    """
    from ..core.gram_schmidt_walk import GramSchmidtWalk
    from ..data.balanced_data_splitter import BalancedDataSplitter
    from ..data.data_processor import DataProcessor
    
    print("=" * 80)
    print("COMPREHENSIVE TOY DATASET TEST SUITE")
    print("=" * 80)
    
    test_suite = create_test_suite()
    results = {}
    
    for test_name, test_data in test_suite.items():
        print(f"\n{'=' * 20} {test_name.upper()} {'=' * 20}")
        print(f"Description: {test_data['description']}")
        print(f"Expected: {test_data['expected_performance']}")
        
        try:
            if 'vectors' in test_data:
                
                vectors = test_data['vectors']
                print(f"Testing {vectors.shape[0]} vectors in {vectors.shape[1]} dimensions")
                
                gsw = GramSchmidtWalk(vectors)
                result = gsw.run(verbose=False)
                
                discrepancy = result['discrepancy']['l_inf']
                print(f"✓ L-inf discrepancy: {discrepancy:.4f}")
                print(f"✓ Iterations: {result['iterations']}")
                print(f"✓ Runtime: {result['time']:.4f} seconds")
                
                
                n = vectors.shape[0]
                theoretical_bound = np.sqrt(np.log(n))
                print(f"✓ Ratio to √log(n): {discrepancy / theoretical_bound:.2f}")
                
                results[test_name] = {
                    'discrepancy': discrepancy,
                    'iterations': result['iterations'],
                    'time': result['time'],
                    'ratio_to_logn': discrepancy / theoretical_bound
                }
                
            elif 'X' in test_data and 'y' in test_data:
                
                X, y = test_data['X'], test_data['y']
                print(f"Testing balanced splitting on {X.shape[0]} samples, {X.shape[1]} features")
                
                splitter = BalancedDataSplitter(validation_size=0.2, random_state=42)
                X_train, X_val, y_train, y_val = splitter.split(X, y, verbose=False)
                
                
                metrics = splitter.evaluate_balance(X, X_train, X_val, y, y_train, y_val)
                
                print(f"✓ Training set: {X_train.shape[0]} samples")
                print(f"✓ Validation set: {X_val.shape[0]} samples")
                print(f"✓ Mean difference improvement: {metrics.get('mean_diff_improvement', 'N/A'):.2f}x")
                print(f"✓ Overall balance score: {metrics.get('overall_balance_score', 'N/A'):.4f}")
                
                orig_ratio = np.mean(y)
                train_ratio = np.mean(y_train)
                val_ratio = np.mean(y_val)
                print(f"✓ Class ratios - Original: {orig_ratio:.3f}, Train: {train_ratio:.3f}, Val: {val_ratio:.3f}")
                
                results[test_name] = {
                    'mean_improvement': metrics.get('mean_diff_improvement', 0),
                    'balance_score': metrics.get('overall_balance_score', 0),
                    'class_ratio_diff_train': abs(orig_ratio - train_ratio),
                    'class_ratio_diff_val': abs(orig_ratio - val_ratio)
                }
                
            elif 'data' in test_data:
                df = test_data['data']
                print(f"Testing mixed types processing on {df.shape[0]} samples, {df.shape[1]} columns")
                
                X = df.drop('target', axis=1)
                y = df['target']
                
                processor = DataProcessor(
                    handle_missing=True,
                    handle_categorical=True,
                    scaling='robust'
                )
                
                balance_vectors = processor.create_balance_vectors(X, y)
                
                print(f"✓ Created {balance_vectors.shape[1]} balance vectors")
                print(f"✓ Max vector norm: {np.max(np.linalg.norm(balance_vectors, axis=1)):.3f}")
                print(f"✓ Data types handled: {X.dtypes.value_counts().to_dict()}")
                print(f"✓ Missing values: {X.isnull().sum().sum()}")
                
                results[test_name] = {
                    'balance_vectors_created': balance_vectors.shape[1],
                    'max_norm': np.max(np.linalg.norm(balance_vectors, axis=1)),
                    'missing_values_handled': X.isnull().sum().sum()
                }
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results[test_name] = {'error': str(e)}
    
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    successful_tests = [name for name, result in results.items() if 'error' not in result]
    failed_tests = [name for name, result in results.items() if 'error' in result]
    
    print(f"✓ Successful tests: {len(successful_tests)}/{len(results)}")
    if failed_tests:
        print(f"✗ Failed tests: {', '.join(failed_tests)}")
    

    vector_tests = {name: result for name, result in results.items() 
                   if 'discrepancy' in result}
    
    if vector_tests:
        best_test = min(vector_tests.items(), key=lambda x: x[1]['discrepancy'])
        worst_test = max(vector_tests.items(), key=lambda x: x[1]['discrepancy'])
        
        print(f"\n📊 Best discrepancy: {best_test[0]} ({best_test[1]['discrepancy']:.4f})")
        print(f"📊 Worst discrepancy: {worst_test[0]} ({worst_test[1]['discrepancy']:.4f})")
    
    return results


def visualize_test_results(results):
    """
    Create visualizations of test suite results.
    
    Args:
        results: Dictionary of test results from run_test_suite()
    """

    vector_results = {name: result for name, result in results.items() 
                     if 'discrepancy' in result}
    
    if not vector_results:
        print("No vector balancing results to visualize")
        return
    

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    names = list(vector_results.keys())
    discrepancies = [vector_results[name]['discrepancy'] for name in names]
    times = [vector_results[name]['time'] for name in names]
    iterations = [vector_results[name]['iterations'] for name in names]
    ratios = [vector_results[name]['ratio_to_logn'] for name in names]
    

    bars1 = ax1.bar(names, discrepancies, color='skyblue', alpha=0.7)
    ax1.set_ylabel('L-inf Discrepancy')
    ax1.set_title('Discrepancy by Test Case')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    

    for bar, val in zip(bars1, discrepancies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom')
    

    bars2 = ax2.bar(names, times, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Runtime by Test Case')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    

    bars3 = ax3.bar(names, iterations, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Iterations')
    ax3.set_title('Iterations by Test Case')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    

    bars4 = ax4.bar(names, ratios, color='gold', alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', label='√log(n) bound')
    ax4.set_ylabel('Ratio to √log(n)')
    ax4.set_title('Performance vs Theoretical Bound')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('toy_dataset_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved as 'toy_dataset_results.png'")


if __name__ == "__main__":

    results = run_test_suite()
    

    visualize_test_results(results)
    

    import json
    with open('toy_dataset_test_results.json', 'w') as f:

        json_results = {}
        for test_name, result in results.items():
            json_results[test_name] = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_results[test_name][key] = value.tolist()
                elif isinstance(value, np.floating):
                    json_results[test_name][key] = float(value)
                elif isinstance(value, np.integer):
                    json_results[test_name][key] = int(value)
                else:
                    json_results[test_name][key] = value
        
        json.dump(json_results, f, indent=2)
    
    print("\nDetailed results saved as 'toy_dataset_test_results.json'")
    print("\n🎉 Test suite complete! Check the generated files for detailed analysis.")