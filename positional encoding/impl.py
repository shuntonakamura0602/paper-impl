import torch
import math

def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            PE[pos, i] = math.sin(angle)
            PE[pos, i + 1] = math.cos(angle)
    return PE

# 例：10単語、512次元
PE = positional_encoding(10, 512)
print(PE.shape)  # torch.Size([10, 512])
