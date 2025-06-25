# paper-impl

## 概要
resnetを論文から実装した。

## ファイル
- model.py - resnetのアーキテクチャ
- train.py - トレーニングループ
- main.py - エントリーポイント

## 学んだこと
- Residual Connectionが勾配消失を解消した理由  
普通のCNNの場合  
$$x_{l+1} = f(W_{l+1})$$