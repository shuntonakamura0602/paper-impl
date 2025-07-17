# 概要
線形回帰(単回帰)モデルをNumpyで一から実装しました

# モデルの式
入力:x
重み:w
バイアス:b
$$y_{pred} = w・x + b$$

# 損失関数(MSE)
$$L = \frac{1}{N}\sum_{i=1}^{N}(y_{true} - y_{pred})^2$$