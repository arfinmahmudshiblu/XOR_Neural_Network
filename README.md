# Key Insights
1. Single Perceptron Limitation: A single perceptron can only create linear decision boundaries. XOR requires a non-linear boundary, which is why it fails.
2. Multi-Layer Solution: By adding a hidden layer, the network can combine multiple linear boundaries to create non-linear decision regions.
3. Hidden Layer Function: Each hidden neuron learns to respond to different patterns in the input space. The output layer then combines these patterns to solve XOR.
4. Universal Approximation: With enough hidden neurons, a 2-layer network can approximate any continuous function, including XOR.

# Exercise: Experiment with the Network
I have tried to modify the following parameters and observe the effects:
Hidden layer size: Try 2, 3, 4, 8 hidden neurons
Learning rate: Try 0.1, 0.5, 1.0, 2.0
Activation functions: Implement ReLU or tanh instead of sigmoid
Initialization: Try different weight initialization strategies

My experimentation code here:
I have tried with 2,3,6,8 hidden neurons hidden_sizes = [2, 3, 4, 8] learning_rates = [0.1, 0.5, 1.0, 2.0] activations = ['sigmoid', 'relu', 'tanh'] inits = ['xavier', 'he', 'normal']
for h in hidden_sizes: for lr in learning_rates: for act in activations: for init in inits: print(f"\nHidden={h}, LR={lr}, Act={act}, Init={init}") model = TwoLayerNetwork(input_size=2, hidden_size=h, output_size=1, activation=act, init=init) losses = model.train(X_train, y_train, epochs=10000, lr=lr) preds = model.predict(X_train)

print("Accuracy with 2,3,4,8 hidden neurons:", print(f"Predictions: {preds.ravel()}, Loss={losses[-1]:.4f}")

# Conclusion
The XOR problem perfectly illustrates why deep learning exists. Simple linear models fail on non-linearly separable data, but adding even one hidden layer with non-linear activation functions enables the network to learn complex patterns. This principle scales up to deep networks with many layers, enabling them to learn incredibly complex functions for tasks like image recognition and natural language processing.


A single-layer perceptron is a linear classifier.It can only learn and classify patterns that are "linearly separable" problem. The XOR problem requires a non-linear decision boundary. A multi-layer neural network with one or more hidden layers can solve the XOR problem. That's why single perceptron cannot solve XOR.
The minimum is 2 hidden neurons in a single hidden layer.
Hidden layers enable non-linear decision boundaries by combining multiple linear transformations with non-linear activation functions, which allows the neural network to learn complex, curved decision surfaces.
