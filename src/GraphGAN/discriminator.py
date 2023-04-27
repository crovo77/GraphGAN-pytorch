import torch
from src.GraphGAN import config


class Discriminator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix: torch.Tensor = torch.randn(self.node_emd_init.shape, dtype=torch.float32)\
            .to(config.torch_device)
        self.bias_vector: torch.Tensor = torch.zeros([self.n_node], dtype=torch.float32).to(config.torch_device)

        self.node_id: torch.Tensor = torch.tensor([0]).to(config.torch_device)
        self.node_neighbor_id: torch.Tensor = torch.tensor([0]).to(config.torch_device)

        self.reward: torch.Tensor = torch.tensor([0], dtype=torch.float32).to(config.torch_device)

        self.node_embedding: torch.Tensor = torch.index_select(self.embedding_matrix, 0,
                                                               self.node_id).to(config.torch_device)
        self.node_neighbor_embedding: torch.Tensor = torch.index_select(self.embedding_matrix, 0,
                                                                        self.node_neighbor_id).to(config.torch_device)
        self.bias: torch.Tensor = torch.index_select(self.bias_vector, 0,
                                                     self.node_neighbor_id).to(config.torch_device)
        self.score: torch.Tensor = self.node_embedding * self.node_neighbor_embedding.sum(0)\
            .to(config.torch_device) + self.bias
