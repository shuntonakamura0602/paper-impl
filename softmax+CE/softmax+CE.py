import numpy as np

# ---------------------------
# Toy data: 3-class, 2D
# ---------------------------
np.random.seed(0)
N = 300
D = 2
C = 3
H = 16

X = np.random.randn(N, D)

# 適当な真の重みでラベル生成（線形分離っぽい）
true_W = np.array([[2.0, -1.0, 0.5],
                   [-1.5, 1.0, 2.0]])
logits_true = X @ true_W
y = np.argmax(logits_true + 0.3*np.random.randn(N, C), axis=1)  # (N,)

# one-hot
Y = np.zeros((N, C))
Y[np.arange(N), y] = 1.0

# ---------------------------
# Model params (2-layer NN)
# ---------------------------
W1 = 0.1 * np.random.randn(D, H)
b1 = np.zeros((1, H))
W2 = 0.1 * np.random.randn(H, C)
b2 = np.zeros((1, C))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    # z: (N, C)
    z_shift = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z_shift)
    return expz / np.sum(expz, axis=1, keepdims=True)

def cross_entropy(probs, y_int):
    # probs: (N, C), y_int: (N,)
    eps = 1e-12
    return -np.mean(np.log(probs[np.arange(len(y_int)), y_int] + eps))

lr = 0.1
for t in range(2000):
    # ---------------------------
    # forward
    # ---------------------------
    Z1 = X @ W1 + b1          # (N, H)
    A1 = relu(Z1)             # (N, H)
    Z2 = A1 @ W2 + b2         # (N, C)  logits
    P = softmax(Z2)           # (N, C)  probs

    loss = cross_entropy(P, y)

    # accuracy (optional)
    if t % 200 == 0:
        pred = np.argmax(P, axis=1)
        acc = np.mean(pred == y)
        print(f"step={t:4d} loss={loss:.4f} acc={acc:.3f}")

    # ---------------------------
    # backward
    # 핵심: softmax + CE -> dZ2 = (P - Y) / N
    # ---------------------------
    dZ2 = (P - Y) / N         # (N, C)
    dW2 = A1.T @ dZ2          # (H, C)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, C)

    dA1 = dZ2 @ W2.T          # (N, H)
    dZ1 = dA1 * (Z1 > 0)      # (N, H)  ReLU'

    dW1 = X.T @ dZ1           # (D, H)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, H)

    # ---------------------------
    # update
    # ---------------------------
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2