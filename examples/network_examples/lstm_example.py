# based on the example here: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3) # Input dim is 3, Output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)] # make a sequence of length 5
print(f'[INPUTS] shape: ({len(inputs)}, {inputs[0].shape}), elements: {inputs}')

hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
print(f'[HIDDEN] shape: ({len(hidden)}, {hidden[0].shape}), elements: {hidden}')
