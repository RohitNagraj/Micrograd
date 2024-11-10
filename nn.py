import random
from engine import Value


class Neuron:
    def __init__(self, n_input):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_input)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, n_input: int, n_outputs: list):
        size = [n_input] + n_outputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    learning_rate = 0.01
    n_iters = 200
    # Sample dataset
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]


    # MSE Loss
    def calculate_loss(y_true, y_pred):
        return sum((y - yp) ** 2 for y, yp in zip(y_true, y_pred))


    n = MLP(3, [4, 4, 1])

    # Training loop
    for iter in range(n_iters):
        # Forward pass
        y_pred = [n(x) for x in xs]
        loss = calculate_loss(ys, y_pred)

        # Backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # Parameter update
        for p in n.parameters():
            p.data -= learning_rate * p.grad

        if iter % 50 == 0:
            print(f"Iter: {iter}, Loss: {loss}")
    final_preds = [n(x) for x in xs]
    print(f"Final Preds: ", final_preds)
