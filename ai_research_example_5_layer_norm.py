#!/usr/bin/env python3
"""
AI Research Example 5: Layer Normalization (2023)
Replicates Layer Normalization, a key component in modern transformers and LLMs

This example demonstrates:
- Layer normalization computation
- Normalization across features for each sample
- Quick simulation to verify the mechanism works correctly

Reference: "Layer Normalization" (Ba et al., 2016) - widely adopted in 2023 LLMs
Application: GPT, BERT, and all modern transformer-based models
"""

import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Apply layer normalization.
    
    Args:
        x: Input tensor (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability
    
    Returns:
        normalized: Layer-normalized output (batch_size, features)
        stats: Dictionary with mean and variance for verification
    """
    # Compute mean and variance across features for each sample
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift
    output = gamma * x_normalized + beta
    
    stats = {
        'mean': mean,
        'variance': variance,
        'x_normalized': x_normalized
    }
    
    return output, stats

def simulate_layer_norm():
    """
    Quick simulation to verify layer normalization works correctly.
    """
    print("=" * 60)
    print("AI Research Example 5: Layer Normalization (2023)")
    print("=" * 60)
    print("\nReplicating: Layer Normalization for LLMs")
    print("Paper: 'Layer Normalization' (Ba et al., 2016)")
    print("Application: GPT-3, GPT-4, Claude, and all modern transformers\n")
    
    # Simple example: 3 samples with 4 features each
    batch_size = 3
    features = 4
    
    print(f"Input dimensions:")
    print(f"  Batch size: {batch_size}")
    print(f"  Features per sample: {features}")
    
    # Create sample input
    np.random.seed(42)
    x = np.random.randn(batch_size, features) * 2 + 5  # Mean ~5, std ~2
    
    # Initialize learnable parameters
    gamma = np.ones(features)
    beta = np.zeros(features)
    
    print(f"\nInput X (before normalization):")
    print(x)
    print(f"\nInput statistics:")
    print(f"  Sample 0 mean: {np.mean(x[0]):.4f}, std: {np.std(x[0]):.4f}")
    print(f"  Sample 1 mean: {np.mean(x[1]):.4f}, std: {np.std(x[1]):.4f}")
    print(f"  Sample 2 mean: {np.mean(x[2]):.4f}, std: {np.std(x[2]):.4f}")
    
    # Apply layer normalization
    output, stats = layer_norm(x, gamma, beta)
    
    print(f"\n--- Layer Normalization Results ---")
    print(f"Output (after normalization):")
    print(output)
    print(f"\nOutput statistics:")
    print(f"  Sample 0 mean: {np.mean(output[0]):.6f}, std: {np.std(output[0]):.6f}")
    print(f"  Sample 1 mean: {np.mean(output[1]):.6f}, std: {np.std(output[1]):.6f}")
    print(f"  Sample 2 mean: {np.mean(output[2]):.6f}, std: {np.std(output[2]):.6f}")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Mean is close to 0 for each sample
    output_means = np.mean(output, axis=-1)
    means_near_zero = np.allclose(output_means, 0, atol=1e-6)
    print(f"✓ Output means close to 0: {means_near_zero}")
    print(f"    Actual means: {output_means}")
    if means_near_zero:
        checks_passed += 1
    
    # Check 2: Std is close to 1 for each sample
    output_stds = np.std(output, axis=-1)
    stds_near_one = np.allclose(output_stds, 1, atol=1e-6)
    print(f"✓ Output std close to 1: {stds_near_one}")
    print(f"    Actual stds: {output_stds}")
    if stds_near_one:
        checks_passed += 1
    
    # Check 3: Output shape matches input shape
    shape_correct = output.shape == x.shape
    print(f"✓ Output shape matches input: {shape_correct}")
    if shape_correct:
        checks_passed += 1
    
    # Check 4: Normalization is stable (no NaN or Inf)
    no_invalid = not (np.any(np.isnan(output)) or np.any(np.isinf(output)))
    print(f"✓ No NaN or Inf values: {no_invalid}")
    if no_invalid:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ Layer normalization works correctly!")
        print("\n💡 Key insight: Layer norm stabilizes training in deep networks")
        print("   by normalizing activations across features for each sample.")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_layer_norm()
    exit(0 if success else 1)
