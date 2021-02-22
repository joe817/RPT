import torch.nn as nn


#class RelationEmbedding(nn.Embedding):
#    def __init__(self, vocab_size, embed_size=512):
#        super().__init__(vocab_size, embed_size, padding_idx=0)

class RelationEmbedding(nn.Parameter):
    def __init__(self, tensors):
        super().__init__(tensors)
#torch.nn.Parameter(torch.zeros(hidden_layer_nodes, output_dimension))
