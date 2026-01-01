#!/usr/bin/env python3
"""
AI Research Example 2: Gradient Descent Optimization
Replicates gradient descent optimization, a fundamental algorithm in deep learning

This example demonstrates:
- Basic gradient descent algorithm
- How parameters are updated iteratively
- Quick simulation to verify convergence on a simple quadratic function
"""

import numpy as np

def quadratic_function(x):
    """
    Simple quadratic function: f(x) = (x - 3)^2 + 1
    Minimum is at x = 3
    """
    return (x - 3) ** 2 + 1

def gradient_quadratic(x):
    """
    Gradient (derivative) of the quadratic function: f'(x) = 2(x - 3)
    """
    return 2 * (x - 3)

def gradient_descent(start_x, learning_rate, num_iterations):
    """
    Perform gradient descent optimization.
    
    Args:
        start_x: Initial parameter value
        learning_rate: Step size for parameter updates
        num_iterations: Number of optimization steps
    
    Returns:
        history: List of (x, f(x)) tuples at each iteration
    """
    x = start_x
    history = [(x, quadratic_function(x))]
    
    for i in range(num_iterations):
        # Compute gradient
        grad = gradient_quadratic(x)
        
        # Update parameter
        x = x - learning_rate * grad
        
        # Record history
        history.append((x, quadratic_function(x)))
    
    return history

def simulate_gradient_descent():
    """
    Quick simulation to verify gradient descent converges correctly.
    """
    print("=" * 60)
    print("AI Research Example 2: Gradient Descent Optimization")
    print("=" * 60)
    print("\nReplicating: Basic Gradient Descent Algorithm")
    print("Fundamental to: All modern deep learning optimization\n")
    
    # Parameters
    start_x = 10.0
    learning_rate = 0.1
    num_iterations = 30
    true_minimum = 3.0
    
    print(f"Optimization parameters:")
    print(f"  Function: f(x) = (x - 3)^2 + 1")
    print(f"  True minimum: x = {true_minimum}")
    print(f"  Starting point: x = {start_x}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {num_iterations}")
    
    # Run gradient descent
    history = gradient_descent(start_x, learning_rate, num_iterations)
    
    print(f"\n--- Optimization Progress ---")
    for i, (x, fx) in enumerate(history[:5]):
        print(f"  Iteration {i}: x = {x:.4f}, f(x) = {fx:.4f}")
    print(f"  ...")
    for i, (x, fx) in enumerate(history[-3:], start=len(history)-3):
        print(f"  Iteration {i}: x = {x:.4f}, f(x) = {fx:.4f}")
    
    final_x, final_fx = history[-1]
    
    print(f"\n--- Results ---")
    print(f"Final x: {final_x:.6f}")
    print(f"Final f(x): {final_fx:.6f}")
    print(f"True minimum x: {true_minimum}")
    print(f"True minimum f(x): {quadratic_function(true_minimum):.6f}")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Converged close to true minimum
    converged = abs(final_x - true_minimum) < 0.01
    print(f"✓ Converged to true minimum (within 0.01): {converged}")
    if converged:
        checks_passed += 1
    
    # Check 2: Function value decreased
    initial_fx = history[0][1]
    decreased = final_fx < initial_fx
    print(f"✓ Function value decreased (from {initial_fx:.4f} to {final_fx:.4f}): {decreased}")
    if decreased:
        checks_passed += 1
    
    # Check 3: Monotonic decrease in later iterations
    last_5_values = [fx for _, fx in history[-5:]]
    monotonic = all(last_5_values[i] >= last_5_values[i+1] - 1e-6 
                    for i in range(len(last_5_values)-1))
    print(f"✓ Monotonic decrease in final iterations: {monotonic}")
    if monotonic:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ Gradient descent works correctly!")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_gradient_descent()
    exit(0 if success else 1)
