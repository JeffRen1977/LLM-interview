#!/usr/bin/env python3
"""
Improved Backpropagation Implementation
Enhanced version with better convergence for XOR problem

This script demonstrates:
1. Improved backpropagation with better hyperparameters
2. Multiple activation functions (sigmoid, tanh, ReLU)
3. Better initialization strategies
4. Enhanced training with momentum
5. Comprehensive evaluation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ImprovedBackpropagation:
    """Improved backpropagation with better convergence"""
    
    def __init__(self, learning_rate=0.5, momentum=0.9, random_seed=42):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # XOR problem data
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        self.y = np.array([[0], [1], [1], [0]], dtype=np.float32)
        
        # Initialize network with better strategy
        self._initialize_network()
        
        # Training history
        self.loss_history = []
        self.epoch_history = []
        
        # Momentum terms
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
    
    def _initialize_network(self):
        """Initialize network with Xavier/Glorot initialization"""
        # Network structure: 2 -> 3 -> 1 (increased hidden layer size)
        self.W1 = np.random.randn(2, 3) * np.sqrt(2.0 / 2)  # Xavier initialization
        self.b1 = np.zeros((1, 3))
        self.W2 = np.random.randn(3, 1) * np.sqrt(2.0 / 3)  # Xavier initialization
        self.b2 = np.zeros((1, 1))
    
    def sigmoid(self, x):
        """Sigmoid activation function with overflow protection"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid activation function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh activation function"""
        return 1 - np.tanh(x) ** 2
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return (x > 0).astype(float)
    
    def forward_propagation(self, X, activation='sigmoid'):
        """Forward propagation through the network"""
        # Layer 1
        z1 = X @ self.W1 + self.b1
        if activation == 'sigmoid':
            a1 = self.sigmoid(z1)
        elif activation == 'tanh':
            a1 = self.tanh(z1)
        elif activation == 'relu':
            a1 = self.relu(z1)
        
        # Layer 2 (output)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)  # Always sigmoid for output
        
        return z1, a1, z2, a2
    
    def compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def backward_propagation(self, X, y, z1, a1, z2, a2, activation='sigmoid'):
        """Backward propagation to compute gradients"""
        m = X.shape[0]
        
        # Output layer gradients
        d_a2 = -(y - a2)
        d_z2 = d_a2 * self.sigmoid_derivative(z2)
        dW2 = a1.T @ d_z2
        db2 = np.sum(d_z2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        d_a1 = d_z2 @ self.W2.T
        if activation == 'sigmoid':
            d_z1 = d_a1 * self.sigmoid_derivative(z1)
        elif activation == 'tanh':
            d_z1 = d_a1 * self.tanh_derivative(z1)
        elif activation == 'relu':
            d_z1 = d_a1 * self.relu_derivative(z1)
        
        dW1 = X.T @ d_z1
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters_with_momentum(self, dW1, db1, dW2, db2):
        """Update network parameters using gradient descent with momentum"""
        # Update momentum terms
        self.vW1 = self.momentum * self.vW1 + self.learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 + self.learning_rate * db1
        self.vW2 = self.momentum * self.vW2 + self.learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 + self.learning_rate * db2
        
        # Update parameters
        self.W1 -= self.vW1
        self.b1 -= self.vb1
        self.W2 -= self.vW2
        self.b2 -= self.vb2
    
    def train(self, epochs=20000, print_interval=2000, activation='sigmoid'):
        """Train the neural network using improved backpropagation"""
        print("🚀 Starting Improved Backpropagation Training")
        print("=" * 60)
        print(f"📊 Network Architecture: 2 -> 3 -> 1")
        print(f"📊 Learning Rate: {self.learning_rate}")
        print(f"📊 Momentum: {self.momentum}")
        print(f"📊 Training Epochs: {epochs}")
        print(f"📊 Activation Function: {activation}")
        print(f"📊 Problem: XOR Logic Gate")
        print("=" * 60)
        
        start_time = time.time()
        best_loss = float('inf')
        patience = 0
        max_patience = 1000
        
        for epoch in range(epochs):
            # Forward propagation
            z1, a1, z2, a2 = self.forward_propagation(self.X, activation)
            
            # Compute loss
            loss = self.compute_loss(self.y, a2)
            
            # Store training history
            if epoch % 100 == 0:
                self.loss_history.append(loss)
                self.epoch_history.append(epoch)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch} (loss: {loss:.6f})")
                    break
            
            # Backward propagation
            dW1, db1, dW2, db2 = self.backward_propagation(self.X, self.y, z1, a1, z2, a2, activation)
            
            # Update parameters with momentum
            self.update_parameters_with_momentum(dW1, db1, dW2, db2)
            
            # Print progress
            if epoch % print_interval == 0:
                print(f"Epoch {epoch:5d}, Loss: {loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Final Loss: {loss:.6f}")
        
        return self.loss_history, self.epoch_history
    
    def predict(self, X=None, activation='sigmoid'):
        """Make predictions using the trained network"""
        if X is None:
            X = self.X
        
        _, _, _, predictions = self.forward_propagation(X, activation)
        return predictions
    
    def evaluate(self, activation='sigmoid'):
        """Evaluate the trained model on XOR problem"""
        print("\n" + "=" * 60)
        print("📈 MODEL EVALUATION")
        print("=" * 60)
        
        predictions = self.predict(activation=activation)
        
        print("Input -> Expected -> Predicted -> Rounded -> Correct")
        print("-" * 55)
        correct = 0
        for i in range(len(self.X)):
            input_val = self.X[i]
            expected = self.y[i][0]
            predicted = predictions[i][0]
            rounded = round(predicted)
            is_correct = rounded == expected
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{input_val} -> {expected:7.0f} -> {predicted:8.4f} -> {rounded:7.0f} -> {status}")
        
        # Calculate accuracy
        accuracy = (correct / len(self.X)) * 100
        print(f"\n🎯 Accuracy: {accuracy:.1f}% ({correct}/{len(self.X)})")
        
        return predictions, accuracy
    
    def plot_training_progress(self, save_plot=True):
        """Plot training loss over epochs"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Loss over epochs
            plt.subplot(2, 1, 1)
            plt.plot(self.epoch_history, self.loss_history, 'b-', linewidth=2)
            plt.title('Improved Backpropagation Training Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Plot 2: Loss over epochs (linear scale)
            plt.subplot(2, 1, 2)
            plt.plot(self.epoch_history, self.loss_history, 'r-', linewidth=2)
            plt.title('Training Loss (Linear Scale)', fontsize=12)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig('improved_backpropagation_training.png', dpi=300, bbox_inches='tight')
                print("📊 Training progress plot saved as 'improved_backpropagation_training.png'")
            
            plt.show()
            
        except ImportError:
            print("📊 Matplotlib not available, skipping plot generation")
    
    def print_network_info(self):
        """Print network architecture and parameters"""
        print("\n" + "=" * 60)
        print("🧠 IMPROVED NETWORK ARCHITECTURE")
        print("=" * 60)
        
        print(f"Input Layer: 2 neurons")
        print(f"Hidden Layer: 3 neurons (sigmoid/tanh/ReLU activation)")
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

def compare_activations():
    """Compare different activation functions"""
    print("🔬 COMPARING ACTIVATION FUNCTIONS")
    print("=" * 60)
    
    activations = ['sigmoid', 'tanh', 'relu']
    results = {}
    
    for activation in activations:
        print(f"\n🧪 Testing {activation.upper()} activation...")
        network = ImprovedBackpropagation(learning_rate=0.5, momentum=0.9)
        network.train(epochs=15000, print_interval=3000, activation=activation)
        predictions, accuracy = network.evaluate(activation=activation)
        results[activation] = accuracy
        
        print(f"📊 {activation.upper()} Accuracy: {accuracy:.1f}%")
    
    print("\n" + "=" * 60)
    print("📈 ACTIVATION FUNCTION COMPARISON")
    print("=" * 60)
    for activation, accuracy in results.items():
        print(f"{activation.upper():8s}: {accuracy:5.1f}%")
    
    best_activation = max(results, key=results.get)
    print(f"\n🏆 Best activation function: {best_activation.upper()} ({results[best_activation]:.1f}%)")
    
    return results

def main():
    """Main function to demonstrate improved backpropagation"""
    print("🧮 Improved Backpropagation Implementation")
    print("=" * 70)
    print("Enhanced version with better convergence, momentum, and multiple activations")
    print("=" * 70)
    
    # Test with sigmoid activation
    print("\n1️⃣  Testing with SIGMOID activation:")
    network = ImprovedBackpropagation(learning_rate=0.5, momentum=0.9)
    network.train(epochs=20000, print_interval=2000, activation='sigmoid')
    predictions, accuracy = network.evaluate(activation='sigmoid')
    network.print_network_info()
    network.plot_training_progress()
    
    # Compare different activation functions
    print("\n2️⃣  Comparing activation functions:")
    results = compare_activations()
    
    print("\n" + "=" * 70)
    print("💡 IMPROVEMENTS DEMONSTRATED")
    print("=" * 70)
    print("1. 🎯 Better Initialization: Xavier/Glorot initialization")
    print("2. 🚀 Momentum: Accelerated convergence with momentum")
    print("3. 🧠 Larger Hidden Layer: 3 neurons instead of 2")
    print("4. 📊 Multiple Activations: Sigmoid, Tanh, ReLU comparison")
    print("5. ⏹️  Early Stopping: Prevent overfitting")
    print("6. 📈 Better Learning Rate: Optimized for XOR problem")
    print("7. 🔍 Comprehensive Evaluation: Detailed accuracy analysis")
    
    print("\n🎉 Improved backpropagation demonstration completed successfully!")
    
    return network, results

if __name__ == "__main__":
    try:
        network, results = main()
    except Exception as e:
        print(f"\n❌ Error during improved backpropagation demo: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure NumPy is installed: pip install numpy")
        print("   • For plotting, install matplotlib: pip install matplotlib")
        print("   • Check if you have sufficient memory for the demo")
