#!/usr/bin/env python3
"""
Tests for AI Research Examples
Validates that all 4 AI research examples run correctly
"""

import unittest
import subprocess
import sys

class TestAIResearchExamples(unittest.TestCase):
    """Test class for AI research example implementations"""
    
    def test_attention_mechanism(self):
        """Test that attention mechanism example runs successfully"""
        result = subprocess.run(
            [sys.executable, 'ai_research_example_1_attention.py'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, 
                        f"Attention example failed: {result.stderr}")
        self.assertIn("Attention mechanism works correctly", result.stdout)
    
    def test_gradient_descent(self):
        """Test that gradient descent example runs successfully"""
        result = subprocess.run(
            [sys.executable, 'ai_research_example_2_gradient_descent.py'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0,
                        f"Gradient descent example failed: {result.stderr}")
        self.assertIn("Gradient descent works correctly", result.stdout)
    
    def test_neural_network(self):
        """Test that neural network example runs successfully"""
        result = subprocess.run(
            [sys.executable, 'ai_research_example_3_neural_network.py'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0,
                        f"Neural network example failed: {result.stderr}")
        self.assertIn("Neural network forward pass works correctly", result.stdout)
    
    def test_softmax_loss(self):
        """Test that softmax and cross-entropy example runs successfully"""
        result = subprocess.run(
            [sys.executable, 'ai_research_example_4_softmax_loss.py'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0,
                        f"Softmax/loss example failed: {result.stderr}")
        self.assertIn("Softmax and cross-entropy work correctly", result.stdout)

if __name__ == "__main__":
    unittest.main()
