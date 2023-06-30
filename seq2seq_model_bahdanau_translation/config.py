import torch

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 35
EMBEDDING_SIZE = 128
ADAM_LR = 0.001
SGD_LR = 0.1
MOMENTUM = 0.9
hidden_size = 128
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")