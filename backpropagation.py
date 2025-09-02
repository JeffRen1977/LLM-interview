#!/usr/bin/env python3
"""
Backpropagation Implementation: Mathematical Principles and Implementation
Extracted from backpropagation.md

This script demonstrates:
1. Mathematical principles of backpropagation using chain rule
2. Pure NumPy implementation of neural network training
3. XOR problem solving with a 2-2-1 network architecture
4. Forward and backward propagation with gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class BackpropagationDemo:
    """Backpropagation demonstration class"""
    
    def __init__(self, learning_rate=0.1, random_seed=42):
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # XOR problem data
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
        self.y = np.array([[0], [1], [1], [0]])              # Expected output (XOR)
        
        # Network structure: 2 -> 2 -> 1
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.zeros((1, 1))
        
        # Training history
        self.loss_history = []
        self.epoch_history = []
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid activation function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        
        # Layer 2 (output)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        
        return z1, a1, z2, a2
    
    def compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def backward_propagation(self, X, y, z1, a1, z2, a2):
        """Backward propagation to compute gradients"""
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        d_a2 = -(y - a2)                                    # dL/da2
        d_z2 = d_a2 * self.sigmoid_derivative(z2)          # dL/dz2
        dW2 = a1.T @ d_z2                                   # dL/dW2
        db2 = np.sum(d_z2, axis=0, keepdims=True)          # dL/db2
        
        # Hidden layer gradients
        d_a1 = d_z2 @ self.W2.T                            # dL/da1
        d_z1 = d_a1 * self.sigmoid_derivative(z1)          # dL/dz1
        dW1 = X.T @ d_z1                                    # dL/dW1
        db1 = np.sum(d_z1, axis=0, keepdims=True)          # dL/db1
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        """Update network parameters using gradient descent"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, epochs=10000, print_interval=1000):
        """Train the neural network using backpropagation"""
        print("🚀 Starting Backpropagation Training")
        print("=" * 50)
        print(f"📊 Network Architecture: 2 -> 2 -> 1")
        print(f"📊 Learning Rate: {self.learning_rate}")
        print(f"📊 Training Epochs: {epochs}")
        print(f"📊 Problem: XOR Logic Gate")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Forward propagation
            z1, a1, z2, a2 = self.forward_propagation(self.X)
            
            # Compute loss
            loss = self.compute_loss(self.y, a2)
            
            # Store training history
            if epoch % 100 == 0:  # Store every 100 epochs for plotting
                self.loss_history.append(loss)
                self.epoch_history.append(epoch)
            
            # Backward propagation
            dW1, db1, dW2, db2 = self.backward_propagation(self.X, self.y, z1, a1, z2, a2)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)
            
            # Print progress
            if epoch % print_interval == 0:
                print(f"Epoch {epoch:5d}, Loss: {loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        return self.loss_history, self.epoch_history
    
    def predict(self, X=None):
        """Make predictions using the trained network"""
        if X is None:
            X = self.X
        
        _, _, _, predictions = self.forward_propagation(X)
        return predictions
    
    def evaluate(self):
        """Evaluate the trained model on XOR problem"""
        print("\n" + "=" * 50)
        print("📈 MODEL EVALUATION")
        print("=" * 50)
        
        predictions = self.predict()
        
        print("Input -> Expected -> Predicted -> Rounded")
        print("-" * 45)
        for i in range(len(self.X)):
            input_val = self.X[i]
            expected = self.y[i][0]
            predicted = predictions[i][0]
            rounded = round(predicted, 3)
            print(f"{input_val} -> {expected:7.0f} -> {predicted:8.4f} -> {rounded:7.3f}")
        
        # Calculate accuracy
        rounded_predictions = np.round(predictions)
        accuracy = np.mean(rounded_predictions == self.y) * 100
        print(f"\n🎯 Accuracy: {accuracy:.1f}%")
        
        return predictions
    
    def plot_training_progress(self):
        """Plot training loss over epochs"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epoch_history, self.loss_history, 'b-', linewidth=2)
            plt.title('Backpropagation Training Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale for better visualization
            plt.tight_layout()
            
            # Save plot
            plt.savefig('backpropagation_training.png', dpi=300, bbox_inches='tight')
            print("📊 Training progress plot saved as 'backpropagation_training.png'")
            plt.show()
            
        except ImportError:
            print("📊 Matplotlib not available, skipping plot generation")
    
    def print_network_info(self):
        """Print network architecture and parameters"""
        print("\n" + "=" * 50)
        print("🧠 NETWORK ARCHITECTURE")
        print("=" * 50)
        
        print(f"Input Layer: 2 neurons")
        print(f"Hidden Layer: 2 neurons (sigmoid activation)")
        print(f"Output Layer: 1 neuron (sigmoid activation)")
        
        print(f"\n📊 Parameter Shapes:")
        print(f"W1: {self.W1.shape} (Input -> Hidden)")
        print(f"b1: {self.b1.shape} (Hidden bias)")
        print(f"W2: {self.W2.shape} (Hidden -> Output)")
        print(f"b2: {self.b2.shape} (Output bias)")
        
        print(f"\n📊 Parameter Values (after training):")
        print(f"W1:\n{self.W1}")
        print(f"b1:\n{self.b1}")
        print(f"W2:\n{self.W2}")
        print(f"b2:\n{self.b2}")

def demonstrate_backpropagation():
    """Main function to demonstrate backpropagation"""
    print("🧮 Backpropagation Mathematical Principles and Implementation")
    print("=" * 70)
    print("This demo implements backpropagation from scratch using pure NumPy")
    print("to solve the XOR problem with a 2-2-1 neural network architecture.")
    print("=" * 70)
    
    # Create and train the network
    network = BackpropagationDemo(learning_rate=0.1, random_seed=42)
    
    # Train the network
    loss_history, epoch_history = network.train(epochs=10000, print_interval=1000)
    
    # Evaluate the model
    predictions = network.evaluate()
    
    # Print network information
    network.print_network_info()
    
    # Plot training progress
    network.plot_training_progress()
    
    print("\n" + "=" * 70)
    print("💡 KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("1. 🔗 Chain Rule: Core mathematical principle of backpropagation")
    print("2. ⚡ Forward Propagation: Computing predictions through the network")
    print("3. 🔄 Backward Propagation: Computing gradients using chain rule")
    print("4. 📉 Gradient Descent: Updating parameters to minimize loss")
    print("5. 🎯 XOR Problem: Non-linearly separable problem requiring hidden layer")
    print("6. 📊 Loss Function: Mean Squared Error for regression-like optimization")
    print("7. 🧠 Activation Function: Sigmoid for non-linear transformations")
    
    print("\n🎉 Backpropagation demonstration completed successfully!")
    
    return network

if __name__ == "__main__":
    try:
        network = demonstrate_backpropagation()
    except Exception as e:
        print(f"\n❌ Error during backpropagation demo: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure NumPy is installed: pip install numpy")
        print("   • For plotting, install matplotlib: pip install matplotlib")
        print("   • Check if you have sufficient memory for the demo")
