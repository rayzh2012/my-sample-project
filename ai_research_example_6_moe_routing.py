#!/usr/bin/env python3
"""
AI Research Example 6: Mixture of Experts (MoE) Routing (2024)
Replicates the routing mechanism in Mixture of Experts models

This example demonstrates:
- Top-K expert selection based on gating scores
- Load balancing across experts
- Quick simulation to verify the routing works correctly

Reference: Used in GPT-4, Mixtral (2024), and other large-scale models
Application: Efficient scaling of model capacity without proportional compute increase
"""

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def moe_routing(inputs, gating_weights, num_experts=8, top_k=2):
    """
    Route inputs to top-K experts using gating network.
    
    Args:
        inputs: Input tokens/samples (batch_size, hidden_dim)
        gating_weights: Gating network weights (hidden_dim, num_experts)
        num_experts: Total number of experts
        top_k: Number of experts to route each token to
    
    Returns:
        expert_indices: Selected expert indices (batch_size, top_k)
        expert_weights: Normalized weights for selected experts (batch_size, top_k)
        load_balance: Count of tokens routed to each expert
    """
    batch_size = inputs.shape[0]
    
    # Compute gating scores
    gating_logits = np.dot(inputs, gating_weights)  # (batch_size, num_experts)
    gating_probs = softmax(gating_logits)
    
    # Select top-K experts for each token
    expert_indices = np.argsort(gating_probs, axis=-1)[:, -top_k:]  # (batch_size, top_k)
    
    # Get weights for selected experts
    expert_weights = np.zeros((batch_size, top_k))
    for i in range(batch_size):
        expert_weights[i] = gating_probs[i, expert_indices[i]]
    
    # Normalize weights across selected experts
    expert_weights = expert_weights / np.sum(expert_weights, axis=-1, keepdims=True)
    
    # Compute load balance (how many tokens assigned to each expert)
    load_balance = np.zeros(num_experts)
    for idx in expert_indices.flatten():
        load_balance[idx] += 1
    
    return expert_indices, expert_weights, load_balance

def simulate_moe_routing():
    """
    Quick simulation to verify MoE routing works correctly.
    """
    print("=" * 60)
    print("AI Research Example 6: Mixture of Experts Routing (2024)")
    print("=" * 60)
    print("\nReplicating: Top-K Expert Selection in MoE Models")
    print("Reference: Mixtral, GPT-4 architecture (2024)")
    print("Application: Efficient scaling of model capacity\n")
    
    # Parameters
    batch_size = 12  # Number of tokens
    hidden_dim = 8
    num_experts = 8
    top_k = 2
    
    print(f"MoE Configuration:")
    print(f"  Batch size (tokens): {batch_size}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of experts: {num_experts}")
    print(f"  Top-K experts per token: {top_k}")
    
    # Create sample inputs and gating weights
    np.random.seed(42)
    inputs = np.random.randn(batch_size, hidden_dim)
    gating_weights = np.random.randn(hidden_dim, num_experts) * 0.1
    
    # Perform routing
    expert_indices, expert_weights, load_balance = moe_routing(
        inputs, gating_weights, num_experts, top_k
    )
    
    print(f"\n--- Routing Results ---")
    print(f"Expert assignments (first 5 tokens):")
    for i in range(min(5, batch_size)):
        experts = expert_indices[i]
        weights = expert_weights[i]
        print(f"  Token {i}: Experts {experts} with weights {weights}")
    
    print(f"\nLoad balance across experts:")
    for i, count in enumerate(load_balance):
        bar = "█" * int(count)
        print(f"  Expert {i}: {int(count):2d} tokens {bar}")
    
    print(f"\nLoad balance statistics:")
    print(f"  Mean: {np.mean(load_balance):.2f}")
    print(f"  Std:  {np.std(load_balance):.2f}")
    print(f"  Min:  {np.min(load_balance):.0f}")
    print(f"  Max:  {np.max(load_balance):.0f}")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Each token assigned to exactly top_k experts
    correct_k = np.all(expert_indices.shape == (batch_size, top_k))
    print(f"✓ Each token assigned to {top_k} experts: {correct_k}")
    if correct_k:
        checks_passed += 1
    
    # Check 2: Expert weights sum to 1 for each token
    weight_sums = np.sum(expert_weights, axis=-1)
    weights_normalized = np.allclose(weight_sums, 1.0)
    print(f"✓ Expert weights normalized (sum to 1): {weights_normalized}")
    if weights_normalized:
        checks_passed += 1
    
    # Check 3: Total load equals batch_size * top_k
    total_load = np.sum(load_balance)
    expected_load = batch_size * top_k
    load_correct = np.isclose(total_load, expected_load)
    print(f"✓ Total load correct ({total_load:.0f} == {expected_load}): {load_correct}")
    if load_correct:
        checks_passed += 1
    
    # Check 4: All weights are positive
    all_positive = np.all(expert_weights > 0)
    print(f"✓ All expert weights are positive: {all_positive}")
    if all_positive:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ MoE routing works correctly!")
        print("\n💡 Key insight: MoE enables scaling model capacity by")
        print("   routing each token to specialized experts, improving")
        print("   efficiency without activating all parameters.")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_moe_routing()
    exit(0 if success else 1)
