#!/usr/bin/env python3
"""
AI Research Example 4: Softmax and Cross-Entropy Loss
Replicates softmax activation and cross-entropy loss for classification

This example demonstrates:
- Softmax activation for probability distribution
- Cross-entropy loss computation
- Quick simulation to verify these work correctly for classification tasks
"""

import numpy as np

def softmax(logits):
    """
    Compute softmax activation to convert logits to probabilities.
    
    Args:
        logits: Raw network outputs (batch_size, num_classes)
    
    Returns:
        probabilities: Softmax probabilities (batch_size, num_classes)
    """
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    """
    Compute cross-entropy loss for classification.
    
    Args:
        predictions: Predicted probabilities (batch_size, num_classes)
        targets: True labels as one-hot vectors (batch_size, num_classes)
    
    Returns:
        loss: Average cross-entropy loss
    """
    batch_size = predictions.shape[0]
    
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Compute cross-entropy
    loss = -np.sum(targets * np.log(predictions_clipped)) / batch_size
    
    return loss

def accuracy(predictions, targets):
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted probabilities (batch_size, num_classes)
        targets: True labels as one-hot vectors (batch_size, num_classes)
    
    Returns:
        acc: Accuracy (fraction of correct predictions)
    """
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return np.mean(predicted_classes == true_classes)

def simulate_classification():
    """
    Quick simulation to verify softmax and cross-entropy work correctly.
    """
    print("=" * 60)
    print("AI Research Example 4: Softmax & Cross-Entropy Loss")
    print("=" * 60)
    print("\nReplicating: Classification Loss Computation")
    print("Fundamental to: All neural network classification tasks\n")
    
    # Parameters
    batch_size = 5
    num_classes = 3
    
    print(f"Classification setup:")
    print(f"  Batch size: {batch_size} samples")
    print(f"  Number of classes: {num_classes}")
    
    # Generate sample logits (raw network outputs)
    np.random.seed(42)
    logits = np.random.randn(batch_size, num_classes)
    
    # Generate sample ground truth labels (one-hot encoded)
    # Use random choice with fixed seed for reproducible results
    true_labels = np.random.choice(num_classes, batch_size)
    targets = np.zeros((batch_size, num_classes))
    targets[np.arange(batch_size), true_labels] = 1
    
    print(f"\nSample logits (raw outputs):")
    print(logits)
    print(f"\nTrue labels: {true_labels}")
    
    # Apply softmax
    probabilities = softmax(logits)
    
    print(f"\n--- Softmax Results ---")
    print(f"Probabilities:")
    for i, (probs, true_label) in enumerate(zip(probabilities, true_labels)):
        print(f"  Sample {i}: {probs} (true class: {true_label})")
    
    # Compute loss
    loss = cross_entropy_loss(probabilities, targets)
    acc = accuracy(probabilities, targets)
    
    print(f"\n--- Loss & Accuracy ---")
    print(f"Cross-entropy loss: {loss:.4f}")
    print(f"Accuracy: {acc:.2%} ({int(acc * batch_size)}/{batch_size} correct)")
    
    # Verification checks
    print(f"\n--- Verification ---")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Probabilities sum to 1
    prob_sums = np.sum(probabilities, axis=1)
    probs_sum_to_one = np.allclose(prob_sums, 1.0)
    print(f"✓ Probabilities sum to 1.0: {probs_sum_to_one}")
    if probs_sum_to_one:
        checks_passed += 1
    
    # Check 2: All probabilities are in [0, 1]
    probs_valid = np.all((probabilities >= 0) & (probabilities <= 1))
    print(f"✓ All probabilities in [0, 1]: {probs_valid}")
    if probs_valid:
        checks_passed += 1
    
    # Check 3: Loss is positive
    loss_positive = loss > 0
    print(f"✓ Loss is positive: {loss_positive}")
    if loss_positive:
        checks_passed += 1
    
    # Check 4: Perfect prediction has lower loss
    # Create perfect predictions
    perfect_predictions = targets.copy()
    perfect_loss = cross_entropy_loss(perfect_predictions, targets)
    perfect_acc = accuracy(perfect_predictions, targets)
    
    print(f"✓ Perfect prediction check:")
    print(f"    Perfect loss: {perfect_loss:.4f} (should be ~0)")
    print(f"    Perfect accuracy: {perfect_acc:.2%} (should be 100%)")
    
    perfect_check = (perfect_loss < 0.01) and (perfect_acc == 1.0)
    if perfect_check:
        checks_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("✓ Softmax and cross-entropy work correctly!")
    else:
        print("⚠ Some checks failed - review implementation")
    print(f"{'='*60}\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = simulate_classification()
    exit(0 if success else 1)
