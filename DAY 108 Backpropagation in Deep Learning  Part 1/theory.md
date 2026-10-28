# Backpropagation in Deep Learning | Part 1 | The What?

## 1. Introduction

Backpropagation is one of the core algorithms used to train artificial neural networks. The word is short for **backward propagation of errors**. It helps a network work out how its trainable parameters—weights and biases—contributed to the error in its predictions.

A neural network starts with initial weights and biases, which usually do not produce good predictions. During training, the network makes predictions, measures its error, calculates gradients, and adjusts its parameters to improve.

## 2. Important terms

- **Input:** The information given to a model.
- **Target / label:** The correct answer the model should learn to predict.
- **Weight:** A trainable value that controls the strength of a connection between neurons.
- **Bias:** A trainable value added to a neuron's weighted input.
- **Forward propagation:** Passing input through the network to calculate a prediction.
- **Prediction:** The network's current output.
- **Loss function:** A mathematical measure of how far predictions are from the targets.
- **Gradient:** A derivative that indicates how a small parameter change affects the loss.
- **Optimizer:** A method that uses gradients to update model parameters. Gradient descent is a common optimizer.
- **Epoch:** One full pass through the training dataset.

## 3. The training loop

A typical training step follows this sequence:

1. **Forward pass:** Compute the network's prediction from the input.
2. **Calculate loss:** Compare the prediction with the correct target.
3. **Backward pass:** Use the chain rule to calculate the loss gradient for each weight and bias.
4. **Parameter update:** An optimizer adjusts the parameters using those gradients.
5. **Repeat:** Continue for many batches or epochs until training is complete or another stopping condition is reached.

## 4. Why is it called “backpropagation”?

The network calculates its prediction by moving forward through its layers. To learn, it starts from the loss at the output and calculates how that loss depends on earlier intermediate values and parameters. This gradient information travels backward through the network, layer by layer.

Backpropagation is not simply “sending the wrong answer backward.” It is a systematic application of calculus that calculates derivatives efficiently.

## 5. The chain rule

The chain rule is a rule from calculus used to differentiate a function made from other functions. Neural networks contain many connected mathematical operations, so the chain rule lets us calculate how a parameter near the beginning of the network affects the final loss.

For a simple chain of functions,

\[
L = f(g(w)),
\]

the derivative is

\[
\frac{dL}{dw} = \frac{dL}{dg}\frac{dg}{dw}.
\]

Backpropagation applies this idea repeatedly across the network.

## 6. A simple intuition

Imagine a network predicts that a picture contains a cat with probability 0.30, while the training label says it is a cat (1). The loss function reports that the prediction is poor. Backpropagation calculates how the weights and biases influenced that loss. The optimizer then changes those parameters so that, ideally, a similar input receives a better prediction next time.

One training example does not guarantee improvement on every later example. Learning happens across many examples and repeated updates.

## 7. Backpropagation versus gradient descent

These terms are related but different:

- **Backpropagation** calculates gradients of the loss with respect to parameters.
- **Gradient descent** uses those gradients to update parameters.
- **The loss function** measures prediction error.
- **Forward propagation** computes predictions.

A common update rule is:

\[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
\]

where \(w\) is a parameter, \(L\) is the loss, and \(\eta\) (eta) is the learning rate. The minus sign moves the parameter in the direction of decreasing loss for ordinary gradient descent.

## 8. What does the learning rate do?

The learning rate controls the size of each update.

- If it is too small, learning may be very slow.
- If it is too large, training may become unstable or fail to settle.
- A suitable value depends on the model, data, and optimizer.

## 9. Where is backpropagation used?

Backpropagation is used to train many kinds of neural networks, including:

- Feed-forward neural networks
- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs)
- Transformer models

The exact network architecture and loss function may differ, but gradient-based learning commonly relies on the same core idea.

## 10. Common misconceptions

1. **“Backpropagation updates the weights by itself.”** More precisely, it calculates gradients; an optimizer uses them to update the weights.
2. **“The network learns from one prediction.”** Usually, training involves many examples and repeated updates.
3. **“The loss must decrease on every step.”** It often trends downward during training, but individual batches or updates can temporarily increase it.
4. **“Backpropagation is the same as forward propagation.”** Forward propagation computes predictions; backpropagation computes gradients from the loss backward through the network.

## 11. Practical notebook

The accompanying `code.ipynb` implements a small neural network with NumPy to learn the XOR pattern. It demonstrates:

- A forward pass through a hidden layer and output layer
- Binary cross-entropy loss
- Gradient calculations using the chain rule
- Gradient-descent parameter updates
- A printed prediction table and a loss curve

### Requirements

Install the libraries if needed:

```bash
python -m pip install numpy matplotlib jupyter
```

Open `code.ipynb` in Jupyter Notebook or VS Code and run the cells from top to bottom.

## 12. Summary

Backpropagation answers the question: **“How did each parameter contribute to the loss?”** It calculates gradients by applying the chain rule backward through a neural network. An optimizer uses those gradients to update the weights and biases, allowing the network to learn from examples.
