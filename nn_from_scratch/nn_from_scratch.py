import numpy as np

# X: (100,2) 入力空間
# W1: (2,10) 2次元→10次元への線形写像
# z1: (100,10) 各サンプルを10次元へ写した結果

# データ（とりあえず適当）
# 100*2の行列を作成
X = np.random.randn(100, 2)
# Xを横方向に足し合わせて、100*1の行列を作成
y = np.sum(X, axis=1, keepdims=True)

# 初期化
# 初期の重みとバイアスを生成
# 2次元空間から10次元空間への線形変換
W1 = np.random.randn(2, 10) * 0.1
b1 = np.zeros((1, 10))

W2 = np.random.randn(10, 1) * 0.1
b2 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

# forward
# 線形変換+平行移動(アフィン変換)
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2

print(z2.shape)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

loss = mse(z2, y)
print(loss)

# 学習率(固定)
lr = 0.01

for _ in range(1000):
    # forward
    # 順伝播
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    # lossの計算
    loss = mse(z2, y)

    # backward
    # 逆伝播
    dz2 = z2 - y
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print("final loss:", loss)