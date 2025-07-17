# 概要
線形回帰(単回帰)モデルをNumpyで一から実装しました

# モデルの式
入力:x
重み:w
バイアス:b
$$y_{pred} = w・x + b$$

# 損失関数(MSE)
$$L = \frac{1}{N}\sum_{i=1}^{N}(y_{true} - y_{pred})^2$$

# 勾配降下法
wの勾配  
$$\frac{\partial L}{\partial w} = - \frac{2}{N} \sum_{i=1}^{N} x_i (y_{true_i} - y_{pred_i})$$  

bの勾配  
$$\frac{\partial L}{\partial b} = - \frac{2}{N} \sum_{i=1}^{N} (y_{true_i} - y_{pred_i})$$
