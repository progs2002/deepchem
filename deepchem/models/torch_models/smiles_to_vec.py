from sre_constants import SRE_FLAG_TEMPLATE
from turtle import forward
from unicodedata import bidirectional
import torch 
import torch.nn as nn 

from typing import List, Dict
from deepchem.utils.typing import OneOrMany

RNN_DICT = {"GRU": nn.GRU, "LSTM": nn.LSTM}

class Smiles2Vec(nn.Module):
    def __init__(self,
                 char_to_idx: Dict[str, int],
                 n_tasks: int = 10,
                 max_seq_len: int = 270,
                 embedding_dim:int = 50,
                 n_classes:int = 2,
                 use_bidir:bool = True,
                 use_conv:bool = True,
                 filters:int = 192,
                 kernel_size:int = 3,
                 strides: int = 1,
                 rnn_sizes: List[int] = [224, 384],
                 rnn_types: List[str] = ["GRU", "GRU"],
                 mode: str = "regression",
                 **kwargs):
        
        super(Smiles2Vec, self).__init__()

        self.char_to_idx = char_to_idx
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.use_bidir = use_bidir
        self.use_conv = use_conv

        self.rnn_types = rnn_types
        self.rnn_sizes = rnn_sizes

        assert len(rnn_sizes) == len(rnn_types),\
              "Should have same number of hidden units as RNNs"
        
        self.n_tasks = n_tasks
        self.mode = mode

        embedding_layer = nn.Embedding(len(self.char_to_idx), self.embedding_dim)
        self.rnn_embedding_layer = nn.Sequential(embedding_layer)

        if use_conv:
            self.kernel_size = kernel_size
            self.filters = filters
            self.strides = strides
            conv_layer = nn.LazyConv1d(out_channels=self.filters,
                               kernel_size=self.kernel_size,
                               stride=self.strides)
            self.rnn_embedding_layer.add_module("conv_layer", conv_layer)

        rnn_modules = [RNN_DICT[rnn_type](self.embedding_dim, rnn_size, bidirectional=self.use_bidir) 
                       for rnn_type, rnn_size in zip(rnn_types, rnn_sizes)]
        self.rnn_layers = nn.Sequential(*rnn_modules)

        if self.mode == "classificaton":
            self.dense_layer = nn.LazyLinear(self.n_tasks * self.n_classes)
        else:
            self.dense_layer = nn.LazyLinear(self.n_tasks)
    
    def forward(self, input: OneOrMany[torch.Tensor]):
        rnn_input = self.rnn_embedding_layer(input)
    
        rnn_embeddings = self.rnn_layers(rnn_input)
        rnn_embeddings = torch.relu(rnn_embeddings)

        if self.mode == "classification":
            logits = self.dense_layer(rnn_embeddings) \
                .reshape(self.n_tasks, self.n_classes)

            if self.n_classes == 2:
                output = torch.sigmoid(logits)
            else:
                output = torch.softmax(logits, self.n_classes)
            
            return [output, logits]
        else:
            output = self.dense_layer(rnn_embeddings) \
                .reshape(self.n_tasks, 1)
            
            return [output]