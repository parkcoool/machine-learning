# Exercise 4-3: Implement computational graph and back-propagation

import random

x_data = [1, 2, 3]
y_data = [2, 4, 6]
w = random.uniform(-10, 10)


def forward(w, x):
    return w * x


def loss(w):
    squares = [(forward(w, x) - y) ** 2 for (x, y) in zip(x_data, y_data)]
    return sum(squares) / len(squares)


print("Prediction before training:", (4, forward(w, 4)))
print("w =", w)

for epoch in range(10):
    for x, y in zip(x_data, y_data):
        y_pred = forward(w, x)
        gradient = 2 * (w * x - y) * x
        w = w - 0.1 * gradient

    print(f"Epoch: {epoch} | Loss: {loss(w)}")

print("Prediction after training:", (4, forward(w, 4)))
print("w =", w)
