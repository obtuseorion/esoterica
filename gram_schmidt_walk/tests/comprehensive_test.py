import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
from ..core.gram_schmidt_walk import GramSchmidtWalk

def create_test_vectors(n, m, distribution="normal", seed=42):
    """
    Create test vectors with various distributions.
    
    Args:
        n: Number of vectors
        m: Dimension of each vector
        distribution: Type of distribution to use
        seed: Random seed
        
    Returns:
        Array of normalized vectors
    """
    np.random.seed(seed)
    
    if distribution == "normal":
        vectors = np.random.normal(0, 1, (n, m))
    elif distribution == "uniform":
        vectors = np.random.uniform(-1, 1, (n, m))
    elif distribution == "spherical":
        vectors = np.random.normal(0, 1, (n, m))
    elif distribution == "adversarial":
        vectors = np.zeros((n, m))
        for i in range(min(n, m)):
            vectors[i, i] = 1.0
        vectors += 0.1 * np.random.normal(0, 1, (n, m))
    elif distribution == "komlós":
        vectors = np.random.normal(0, 1, (n, m))
        for j in range(m):
            vectors[:, j] -= vectors[:, j].mean()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / np.maximum(norms[:, np.newaxis], 1.0)

def analyze_subgaussianity(result, vectors, confidence=0.95):
    """
    Analyze the subgaussianity of the resulting discrepancy vector.
    The paper claims O(1)-subgaussianity for all directions.
    
    Args:
        result: Result dictionary from GramSchmidtWalk.run()
        vectors: Original vectors used in the algorithm
        confidence: Confidence level for statistical tests
        
    Returns:
        Dictionary with subgaussianity analysis metrics
    """
    coloring = result["coloring"]
    discrepancy_vector = vectors.T @ coloring
    
    num_directions = 1000
    directions = np.random.normal(0, 1, (num_directions, vectors.shape[1]))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    projections = directions @ discrepancy_vector
    
    sigma_est = np.std(projections)
    
    ks_statistic, p_value = stats.kstest(projections / sigma_est, 'norm')
    
    ci_width = np.sqrt(2 * np.log(2 / (1 - confidence)))
    expected_within_ci = confidence
    actual_within_ci = np.mean(np.abs(projections) <= ci_width * sigma_est)
    
    max_proj = np.max(np.abs(projections))
    subg_param = np.sqrt(2 * np.log(max_proj / sigma_est))
    
    return {
        "sigma": sigma_est,
        "ks_statistic": ks_statistic,
        "p_value": p_value,
        "expected_within_ci": expected_within_ci,
        "actual_within_ci": actual_within_ci,
        "subgaussian_parameter": subg_param,
        "max_projection": max_proj
    }

def compare_to_random_coloring(vectors, result, num_trials=100):
    """
    Compare the discrepancy of our result to random colorings.
    
    Args:
        vectors: Original vectors used in the algorithm
        result: Result dictionary from GramSchmidtWalk.run()
        num_trials: Number of random colorings to generate
        
    Returns:
        Dictionary with comparison metrics
    """
    n = vectors.shape[0]
    gsw_discrepancy = result["discrepancy"]["l_inf"]
    
    random_discrepancies = []
    for _ in range(num_trials):
        random_coloring = np.random.choice([-1, 1], size=n)
        combined = vectors.T @ random_coloring
        random_discrepancies.append(np.max(np.abs(combined)))
    
    percentile = stats.percentileofscore(random_discrepancies, gsw_discrepancy)
    
    return {
        "gsw_discrepancy": gsw_discrepancy,
        "random_mean": np.mean(random_discrepancies),
        "random_min": np.min(random_discrepancies),
        "random_max": np.max(random_discrepancies),
        "percentile": percentile,
        "improvement_ratio": np.mean(random_discrepancies) / gsw_discrepancy
    }

def compare_to_theoretical_bound(n, m, result):
    """
    Compare our result to theoretical bounds from the paper.
    
    Args:
        n: Number of vectors
        m: Dimension of each vector
        result: Result from GramSchmidtWalk.run()
        
    Returns:
        Dictionary with comparison metrics
    """
    paper_bound = np.sqrt(40)

    logn_bound = np.sqrt(np.log(n))
    logm_bound = np.sqrt(np.log(m))
    lognm_bound = np.sqrt(np.log(n*m))
    
    actual_discrepancy = result["discrepancy"]["l_inf"]
    
    return {
        "actual_discrepancy": actual_discrepancy,
        "paper_bound": paper_bound,
        "logn_bound": logn_bound,
        "logm_bound": logm_bound,
        "lognm_bound": lognm_bound,
        "ratio_to_paper": actual_discrepancy / paper_bound,
        "ratio_to_logn": actual_discrepancy / logn_bound,
        "ratio_to_logm": actual_discrepancy / logm_bound,
        "ratio_to_lognm": actual_discrepancy / lognm_bound
    }

def plot_discrepancy_evolution(result):
    """
    Plot how the discrepancy evolves during the algorithm.
    
    Args:
        result: Result dictionary from GramSchmidtWalk.run()
    """
    history = result["discrepancy_history"]
    iterations = list(range(1, len(history) + 1))
    
    l_inf_values = [d["l_inf"] for d in history]
    l2_values = [d["l2"] for d in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, l_inf_values, label="L-inf discrepancy")
    plt.plot(iterations, l2_values, label="L2 discrepancy")
    plt.axhline(y=np.sqrt(40), color='r', linestyle='--', label="Paper bound (√40 ≈ 6.32)")

    plt.axhline(y=np.sqrt(np.log(len(iterations))), color='g', linestyle='--', 
                label=f"√log(n) ≈ {np.sqrt(np.log(len(iterations))):.2f}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Discrepancy")
    plt.title("Evolution of Discrepancy During Gram-Schmidt Walk")
    plt.legend()
    plt.grid(True)
    plt.savefig("discrepancy_evolution.png")
    plt.close()

def plot_projection_distribution(result, vectors):
    """
    Plot the distribution of random projections to verify subgaussianity.
    
    Args:
        result: Result dictionary from GramSchmidtWalk.run()
        vectors: Original vectors used in the algorithm
    """
    coloring = result["coloring"]
    discrepancy_vector = vectors.T @ coloring
    
    num_directions = 1000
    directions = np.random.normal(0, 1, (num_directions, vectors.shape[1]))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    projections = directions @ discrepancy_vector

    plt.figure(figsize=(10, 6))

    sigma = np.std(projections)
    normalized_projections = projections / sigma

    plt.hist(normalized_projections, bins=30, density=True, alpha=0.7, label="Projections")

    x = np.linspace(-4, 4, 1000)
    plt.plot(x, stats.norm.pdf(x), 'r', label="Standard Normal")

    for k in [1, 2, 3]:
        bound = np.sqrt(2 * np.log(2 / (1 - stats.norm.cdf(k))))
        plt.axvline(x=bound, color='g', linestyle='--', 
                    label=f"{k}-sigma bound: {bound:.2f}")
        plt.axvline(x=-bound, color='g', linestyle='--')
    
    plt.xlabel("Normalized Projection Value")
    plt.ylabel("Density")
    plt.title("Distribution of Random Projections (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.savefig("projection_distribution.png")
    plt.close()

def print_detailed_report(n, m, distribution, result, subg_analysis, comparison, theoretical):
    """Print a detailed report of the test results"""
    print("\n" + "="*80)
    print(f"DETAILED REPORT FOR {distribution.upper()} DISTRIBUTION (n={n}, m={m})")
    print("="*80)

    print(f"Vectors: {n} vectors in {m} dimensions")
    print(f"Algorithm completed in {result['time']:.4f} seconds and {result['iterations']} iterations")

    print("\nDISCREPANCY RESULTS:")
    print(f"L-inf discrepancy: {result['discrepancy']['l_inf']:.4f}")
    print(f"L2 discrepancy: {result['discrepancy']['l2']:.4f}")
    print(f"L1 discrepancy: {result['discrepancy']['l1']:.4f}")

    print("\nSUBGAUSSIANITY ANALYSIS:")
    print(f"Estimated sigma: {subg_analysis['sigma']:.4f}")
    print(f"Subgaussian parameter: {subg_analysis['subgaussian_parameter']:.4f}")
    print(f"Maximum projection: {subg_analysis['max_projection']:.4f}")
    print(f"KS test p-value: {subg_analysis['p_value']:.4f}")
    print(f"Expected vs actual within CI: {subg_analysis['expected_within_ci']:.2f} vs {subg_analysis['actual_within_ci']:.2f}")

    print("\nCOMPARISON TO RANDOM COLORINGS:")
    print(f"GSW discrepancy: {comparison['gsw_discrepancy']:.4f}")
    print(f"Random coloring (mean): {comparison['random_mean']:.4f}")
    print(f"Random coloring (min/max): {comparison['random_min']:.4f} / {comparison['random_max']:.4f}")
    print(f"Percentile in random distribution: {comparison['percentile']:.2f}%")
    print(f"Improvement ratio: {comparison['improvement_ratio']:.2f}x better than random")

    print("\nCOMPARISON TO THEORETICAL BOUNDS:")
    print(f"Paper bound (√40): {theoretical['paper_bound']:.4f}")
    print(f"Ratio to paper bound: {theoretical['ratio_to_paper']:.4f}")
    print(f"√log(n) bound: {theoretical['logn_bound']:.4f}")
    print(f"√log(m) bound: {theoretical['logm_bound']:.4f}")
    print(f"√log(nm) bound: {theoretical['lognm_bound']:.4f}")
    
    print("\nPLOTS GENERATED:")
    print("- discrepancy_evolution.png: Shows how discrepancy evolves during iterations")
    print("- projection_distribution.png: Distribution of random projections")
    
    print("="*80 + "\n")

def run_comprehensive_test(n, m, distribution="normal", seed=42):
    """
    Run a comprehensive test of the Gram-Schmidt Walk algorithm.
    
    Args:
        n: Number of vectors
        m: Dimension of each vector
        distribution: Type of distribution to use
        seed: Random seed
        
    Returns:
        Dictionary with test results
    """
    print(f"Running comprehensive test with {n} vectors in {m} dimensions, {distribution} distribution...")

    vectors = create_test_vectors(n, m, distribution=distribution, seed=seed)

    gsw = GramSchmidtWalk(vectors)
    result = gsw.run(verbose=True)

    subg_analysis = analyze_subgaussianity(result, vectors)
    comparison = compare_to_random_coloring(vectors, result)
    theoretical = compare_to_theoretical_bound(n, m, result)

    plot_discrepancy_evolution(result)
    plot_projection_distribution(result, vectors)
    print_detailed_report(n, m, distribution, result, subg_analysis, comparison, theoretical)
    
    return {
        "vectors": vectors,
        "result": result,
        "subgaussianity": subg_analysis,
        "random_comparison": comparison,
        "theoretical_comparison": theoretical
    }

def run_scaling_test(max_size=500, step=50, dimension_ratio=0.2):
    """
    Test how the algorithm scales with problem size.
    
    Args:
        max_size: Maximum number of vectors to test
        step: Step size for increasing the number of vectors
        dimension_ratio: Ratio of dimensions to vectors
        
    Returns:
        Dictionary with test results
    """
    sizes = list(range(step, max_size + step, step))
    results = {}
    
    print("Running scaling tests...")
    
    for n in sizes:
        m = max(2, int(n * dimension_ratio))
        
        print(f"Testing with n={n}, m={m}...")
        
        vectors = create_test_vectors(n, m)
        
        start_time = time.time()
        gsw = GramSchmidtWalk(vectors)
        result = gsw.run()
        elapsed_time = time.time() - start_time
        
        theoretical = compare_to_theoretical_bound(n, m, result)
        
        results[n] = {
            "n": n,
            "m": m,
            "time": elapsed_time,
            "iterations": result["iterations"],
            "l_inf_discrepancy": result["discrepancy"]["l_inf"],
            "l2_discrepancy": result["discrepancy"]["l2"],
            "theoretical": theoretical
        }
        
        print(f"  Completed in {elapsed_time:.4f} seconds")
        print(f"  L-inf discrepancy: {result['discrepancy']['l_inf']:.4f}")

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot([r["n"] for r in results.values()], [r["time"] for r in results.values()], 'o-')
    plt.xlabel("Number of vectors (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot([r["n"] for r in results.values()], [r["iterations"] for r in results.values()], 'o-')
    plt.xlabel("Number of vectors (n)")
    plt.ylabel("Iterations")
    plt.title("Iteration Count Scaling")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot([r["n"] for r in results.values()], 
             [r["l_inf_discrepancy"] for r in results.values()], 'bo-', label="L-inf")
    plt.plot([r["n"] for r in results.values()], 
             [r["theoretical"]["paper_bound"] for r in results.values()], 'r--', 
             label="Paper bound (√40)")
    plt.plot([r["n"] for r in results.values()], 
             [r["theoretical"]["logn_bound"] for r in results.values()], 'g--', 
             label="√log(n)")
    plt.xlabel("Number of vectors (n)")
    plt.ylabel("Discrepancy")
    plt.title("L-inf Discrepancy Scaling")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot([r["n"] for r in results.values()], 
             [r["theoretical"]["ratio_to_paper"] for r in results.values()], 'ro-', 
             label="Ratio to paper bound")
    plt.plot([r["n"] for r in results.values()], 
             [r["theoretical"]["ratio_to_logn"] for r in results.values()], 'go-', 
             label="Ratio to √log(n)")
    plt.xlabel("Number of vectors (n)")
    plt.ylabel("Ratio")
    plt.title("Ratio to Theoretical Bounds")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("scaling_results.png")
    plt.close()
    
    print("\nScaling test results:")
    print("="*80)
    print(f"{'n':>5} {'m':>5} {'Time (s)':>10} {'Iterations':>10} {'L-inf':>8} {'√log(n)':>8} {'Ratio':>8}")
    print("-"*80)
    
    for n, r in results.items():
        print(f"{r['n']:5d} {r['m']:5d} {r['time']:10.4f} {r['iterations']:10d} {r['l_inf_discrepancy']:8.4f} {r['theoretical']['logn_bound']:8.4f} {r['theoretical']['ratio_to_logn']:8.4f}")
    
    print("="*80)
    print("Plot saved to scaling_results.png")
    
    return results

if __name__ == "__main__":
    test_result = run_comprehensive_test(n=100, m=20, distribution="normal")

    distributions = ["uniform", "komlós", "adversarial"]
    for dist in distributions:
        run_comprehensive_test(n=100, m=20, distribution=dist)

    scaling_results = run_scaling_test(max_size=300, step=50)