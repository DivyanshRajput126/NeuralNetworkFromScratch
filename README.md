# Neural Network from Scratch 🧠

A simple **feedforward neural network** built from scratch using **Python** and **NumPy**. This project demonstrates the core principles behind deep learning, including **forward propagation**, **backpropagation**, and **gradient descent**.

---

## 🚀 Features
✅ No external deep learning libraries like TensorFlow or PyTorch—just **pure Python + NumPy**  
✅ Supports multiple layers and activation functions  
✅ Implements **Stochastic Gradient Descent (SGD)** for optimization  
✅ Works for **classification tasks** (e.g., predicting handwritten digits from MNIST)  
✅ Easy to extend and modify  


## ⚙️ How It Works
### **1️⃣ Forward Propagation**
The input data is passed through multiple layers, applying weights, biases, and activation functions to compute the output.

### **2️⃣ Loss Calculation**
The difference between the predicted and actual values is measured using a loss function (e.g., Mean Squared Error for regression, Cross-Entropy for classification).

### **3️⃣ Backpropagation**
The error is propagated backward, computing gradients for each layer using **partial derivatives**.

### **4️⃣ Gradient Descent**
The gradients are used to update the weights and biases, minimizing the loss over multiple iterations.

---

## 📜 Mathematical Explanation
The weight updates are calculated using the formula:

\[
W = W - \alpha \frac{\partial L}{\partial W}
\]

Where:  
- \( W \) = Weights  
- \( \alpha \) = Learning Rate  
- \( L \) = Loss Function  

Each layer computes:

\[
Z = W \cdot X + B
\]

and applies an **activation function**:

\[
A = \sigma(Z)
\]

---

## 🛠 Copy all codebooks
1. **Clone the repository**
   ```sh
   git clone https://github.com/DivyanshRajput126/NeuralNetworkFromScratch.git
   cd NeuralNetworkFromScratch
   ````
---

## 📊 Example Usage
```python
from neural_network import NeuralNetwork

# Define a neural network with 3 layers (input, hidden, output)
nn = NeuralNetwork([2, 4, 1], activation="sigmoid")

# Train the model with sample data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn.train(X, y, epochs=1000, learning_rate=0.1)

# Make predictions
print(nn.predict(X))
```

---

## 🧩 Supported Activation Functions
✅ **Sigmoid**  
✅ **ReLU**  
✅ **Softmax**  

---

## 🎯 Future Improvements
- 🏆 Add support for batch training  
- 🎨 Implement different optimizers and activation functions(SGD,AdaDelta,TanH)  
- 📈 Visualize training loss over epochs  
- 🔥 Extend to deeper architectures  

---

## 💡 Contributing
Want to improve this project? Contributions are welcome! 🎉  
- Fork the repo  
- Create a new branch  
- Submit a pull request 🚀  

---

## 📜 License
MIT License. Feel free to use, modify, and distribute. 😊  

---
