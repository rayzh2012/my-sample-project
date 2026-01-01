#!/usr/bin/env python3
"""
AI Research Example 1: Attention Mechanism
Replicates the core attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017)

This example demonstrates:
- Scaled Dot-Product Attention
- How attention weights are computed
- Quick simulation to verify the mechanism works correctly
"""

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
    
    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Compute weighted sum of values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def simulate_attention():
    """
    Quick simulation to verify attention mechanism works correctly.
    """
    print("=" * 60)
    print("AI Research Example 1: Attention Mechanism")
    print("=" * 60)
    print("\nReplicating: Scaled Dot-Product Attention from Transformers")
    print("Paper: 'Attention Is All You Need' (Vaswani et al., 2017)\n")
    
    # Simple example: 3 words with 4-dimensional embeddings
    seq_len = 3
    d_model = 4
    
    # Create sample Query, Key, Value matrices
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)
    
    print(f"Input dimensions:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {d_model}")
    print(f"\nQuery matrix Q shape: {Q.shape}")
    print(f"Key matrix K shape: {K.shape}")
    print(f"Value matrix V shape: {V.shape}")
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\n--- Results ---")
    print(f"\nAttention weights (each row sums to 1.0):")
    for i, row in enumerate(attention_weights):
        print(f"  Position {i}: {row} (sum={np.sum(row):.4f})")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Attention weights sum to 1
    weights_sum_correct = np.allclose(np.sum(attention_weights, axis=1), 1.0)
    print(f"✓ Attention weights sum to 1.0: {weights_sum_correct}")
    if weights_sum_correct:
        checks_passed += 1
    
    # Check 2: Output dimensions match expected
    output_shape_correct = output.shape == (seq_len, d_model)
    print(f"✓ Output shape is correct: {output_shape_correct}")
    if output_shape_correct:
        checks_passed += 1
    
    # Check 3: Attention weights are all positive
    weights_positive = np.all(attention_weights >= 0)
    print(f"✓ Attention weights are non-negative: {weights_positive}")
    if weights_positive:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ Attention mechanism works correctly!")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_attention()
    exit(0 if success else 1)
