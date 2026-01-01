#!/usr/bin/env python3
"""
AI Research Example 3: Neural Network Forward Pass
Replicates a simple feedforward neural network forward pass

This example demonstrates:
- Multi-layer neural network structure
- Forward propagation through layers
- Activation functions (ReLU)
- Quick simulation to verify computations work correctly
"""

import numpy as np

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    ReLU(x) = max(0, x)
    """
    return np.maximum(0, x)

class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network.
    Architecture: input -> hidden layer (ReLU) -> output layer
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network with random weights.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
        """
        # Initialize weights with small random values
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data (batch_size, input_size)
        
        Returns:
            output: Network output (batch_size, output_size)
            cache: Intermediate values for verification
        """
        # Layer 1: Linear transformation + ReLU
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        
        # Layer 2: Linear transformation
        z2 = np.dot(a1, self.W2) + self.b2
        output = z2
        
        # Cache intermediate values
        cache = {
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'output': output
        }
        
        return output, cache

def simulate_neural_network():
    """
    Quick simulation to verify neural network forward pass works correctly.
    """
    print("=" * 60)
    print("AI Research Example 3: Neural Network Forward Pass")
    print("=" * 60)
    print("\nReplicating: Feedforward Neural Network Computation")
    print("Fundamental to: All deep learning architectures\n")
    
    # Network architecture
    input_size = 4
    hidden_size = 8
    output_size = 3
    batch_size = 2
    
    print(f"Network architecture:")
    print(f"  Input layer: {input_size} units")
    print(f"  Hidden layer: {hidden_size} units (ReLU activation)")
    print(f"  Output layer: {output_size} units")
    print(f"  Batch size: {batch_size} samples")
    
    # Create network
    network = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    
    print(f"\nWeight matrices:")
    print(f"  W1 shape: {network.W1.shape}")
    print(f"  W2 shape: {network.W2.shape}")
    
    # Create sample input
    X = np.random.randn(batch_size, input_size)
    print(f"\nInput X shape: {X.shape}")
    
    # Forward pass
    output, cache = network.forward(X)
    
    print(f"\n--- Forward Pass Results ---")
    print(f"Hidden layer activation (a1) shape: {cache['a1'].shape}")
    print(f"Hidden layer activation (a1) sample:\n{cache['a1'][0][:4]}...")
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Output shape is correct
    output_shape_correct = output.shape == (batch_size, output_size)
    print(f"✓ Output shape is correct: {output_shape_correct}")
    if output_shape_correct:
        checks_passed += 1
    
    # Check 2: Hidden activation shape is correct
    hidden_shape_correct = cache['a1'].shape == (batch_size, hidden_size)
    print(f"✓ Hidden activation shape is correct: {hidden_shape_correct}")
    if hidden_shape_correct:
        checks_passed += 1
    
    # Check 3: ReLU activation works (all values >= 0)
    relu_works = np.all(cache['a1'] >= 0)
    print(f"✓ ReLU activation works (all values >= 0): {relu_works}")
    if relu_works:
        checks_passed += 1
    
    # Check 4: Output is not all zeros (network is learning something)
    not_all_zeros = not np.allclose(output, 0)
    print(f"✓ Network produces non-zero output: {not_all_zeros}")
    if not_all_zeros:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ Neural network forward pass works correctly!")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_neural_network()
    exit(0 if success else 1)
