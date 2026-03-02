## 計算過程
$$L=\frac{1}{2N}\sum_{i=1}^{N}(z_{2,i}-y_{i})^2$$
1サンプルだけ取り出すと
$$L_{i}=\frac{1}{2}(z_{2,i}-y_{i})^2$$
これをz2,i微分すると
$$\frac{\partial Li}{\partial z_{2,i}}=(z_{2,i}-y_{i})$$
全体の損失は平均しているので、
$$\frac{\partial Li}{\partial z_{2,i}}=\frac{1}{N}(z_{2,i}-y_{i})$$

### dW2 = a1.T @ dz2 / N の導出
$$z_{2,i}=\sum_{j=1}^{N}a_{1,i,j}W_{2,j}+b_{2}$$
W2jについての勾配は
$$\frac{\partial L}{\partial W_{2,j}}=\sum_{i=1}^{N}\frac{\partial L}{\partial z_{2,i}}\frac{\partial z_{2,i}}{\partial W_{2,j}}=\sum_{i=1}^{N}\frac{1}{N}(z_{2,i}-y_{i})a_{1,i,j}$$
$$\frac{\partial L}{\partial W_{2}}=\frac{1}{N}A_{1}^T(Z_{2}-Y)$$
これがそのままコードになっている   
```python
dW2 = a1.T @ dz2 / len(X)
```