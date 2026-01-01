#!/usr/bin/env python3
"""
AI Research Example 7: Rotary Position Embeddings (RoPE) (2025)
Replicates Rotary Position Embeddings used in modern LLMs

This example demonstrates:
- Rotary position encoding mechanism
- Position-aware attention through rotation
- Quick simulation to verify the mechanism works correctly

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
Application: LLaMA, GPT-NeoX, PaLM, and most 2024-2025 open-source LLMs
"""

import numpy as np

def get_rotary_matrix(seq_len, dim, base=10000):
    """
    Generate rotary position embedding matrix.
    
    Args:
        seq_len: Sequence length
        dim: Embedding dimension (must be even)
        base: Base for computing frequencies
    
    Returns:
        cos: Cosine components (seq_len, dim)
        sin: Sine components (seq_len, dim)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    
    # Generate position indices
    positions = np.arange(seq_len)
    
    # Compute angles: outer product of positions and inverse frequencies
    angles = np.outer(positions, inv_freq)  # (seq_len, dim//2)
    
    # Duplicate to match full dimension
    angles = np.repeat(angles, 2, axis=-1)  # (seq_len, dim)
    
    # Compute cos and sin
    cos = np.cos(angles)
    sin = np.sin(angles)
    
    return cos, sin

def apply_rotary_embedding(x, cos, sin):
    """
    Apply rotary position embedding to input.
    
    Args:
        x: Input tensor (batch_size, seq_len, dim)
        cos: Cosine components (seq_len, dim)
        sin: Sine components (seq_len, dim)
    
    Returns:
        x_rotated: Position-encoded tensor (batch_size, seq_len, dim)
    """
    batch_size, seq_len, dim = x.shape
    
    # Split into even and odd indices (pairs)
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    cos_even = cos[..., 0::2]
    sin_even = sin[..., 0::2]
    
    # Apply rotation formula: proper 2D rotation for each pair
    # For 2D rotation: (x1', x2') = (x1*cos - x2*sin, x1*sin + x2*cos)
    rotated1 = x1 * cos_even - x2 * sin_even
    rotated2 = x1 * sin_even + x2 * cos_even
    
    # Interleave back
    x_rotated = np.zeros_like(x)
    x_rotated[..., 0::2] = rotated1
    x_rotated[..., 1::2] = rotated2
    
    return x_rotated

def simulate_rope():
    """
    Quick simulation to verify RoPE works correctly.
    """
    print("=" * 60)
    print("AI Research Example 7: Rotary Position Embeddings (2025)")
    print("=" * 60)
    print("\nReplicating: RoPE for Position-Aware Attention")
    print("Paper: 'RoFormer' (Su et al., 2021)")
    print("Application: LLaMA, Mistral, Qwen, and most modern LLMs\n")
    
    # Parameters
    batch_size = 2
    seq_len = 5
    dim = 8  # Must be even
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {dim}")
    
    # Create sample input
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, dim)
    
    print(f"\nInput tensor shape: {x.shape}")
    print(f"Sample input (first sequence, first position):")
    print(f"  {x[0, 0]}")
    
    # Generate rotary embeddings
    cos, sin = get_rotary_matrix(seq_len, dim)
    
    print(f"\n--- Rotary Matrix Components ---")
    print(f"Cosine matrix shape: {cos.shape}")
    print(f"Sine matrix shape: {sin.shape}")
    print(f"\nCosine values at position 0: {cos[0]}")
    print(f"Sine values at position 0: {sin[0]}")
    
    # Apply rotary embeddings
    x_rotated = apply_rotary_embedding(x, cos, sin)
    
    print(f"\n--- Results ---")
    print(f"Output tensor shape: {x_rotated.shape}")
    print(f"Sample output (first sequence, first position):")
    print(f"  {x_rotated[0, 0]}")
    
    # Check relative position encoding
    # Compute similarity before and after rotation for different positions
    q = x[0, 0]  # Query at position 0
    k1 = x[0, 0]  # Key at same position
    k2 = x[0, 2]  # Key at different position
    
    q_rot = x_rotated[0, 0]
    k1_rot = x_rotated[0, 0]
    k2_rot = x_rotated[0, 2]
    
    sim_before_same = np.dot(q, k1)
    sim_after_same = np.dot(q_rot, k1_rot)
    sim_before_diff = np.dot(q, k2)
    sim_after_diff = np.dot(q_rot, k2_rot)
    
    print(f"\nPosition-aware similarity:")
    print(f"  Before RoPE - Same position: {sim_before_same:.4f}")
    print(f"  After RoPE  - Same position: {sim_after_same:.4f}")
    print(f"  Before RoPE - Diff position: {sim_before_diff:.4f}")
    print(f"  After RoPE  - Diff position: {sim_after_diff:.4f}")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Output shape matches input shape
    shape_correct = x_rotated.shape == x.shape
    print(f"✓ Output shape matches input: {shape_correct}")
    if shape_correct:
        checks_passed += 1
    
    # Check 2: Magnitude is approximately preserved (rotation property)
    input_norms = np.linalg.norm(x, axis=-1)
    output_norms = np.linalg.norm(x_rotated, axis=-1)
    norms_preserved = np.allclose(input_norms, output_norms, rtol=1e-5)
    print(f"✓ Vector magnitudes preserved: {norms_preserved}")
    print(f"    Sample input norm:  {input_norms[0, 0]:.6f}")
    print(f"    Sample output norm: {output_norms[0, 0]:.6f}")
    if norms_preserved:
        checks_passed += 1
    
    # Check 3: No NaN or Inf values
    no_invalid = not (np.any(np.isnan(x_rotated)) or np.any(np.isinf(x_rotated)))
    print(f"✓ No NaN or Inf values: {no_invalid}")
    if no_invalid:
        checks_passed += 1
    
    # Check 4: Same position maintains high similarity after rotation
    same_pos_maintained = np.isclose(sim_after_same, sim_before_same, rtol=0.1)
    print(f"✓ Same position similarity maintained: {same_pos_maintained}")
    if same_pos_maintained:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ Rotary Position Embeddings work correctly!")
        print("\n💡 Key insight: RoPE encodes position information through")
        print("   rotation, enabling relative position awareness without")
        print("   explicit position embeddings. Used in most modern LLMs!")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_rope()
    exit(0 if success else 1)
