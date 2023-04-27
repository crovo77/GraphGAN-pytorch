import torch
from src import utils
from src.GraphGAN import config


class Generator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.embedding_matrix: torch.Tensor = torch.randn(self.node_emd_init.shape, dtype=torch.float32)\
            .to(config.torch_device)
        self.bias_vector: torch.Tensor = torch.zeros([self.n_node], dtype=torch.float32).to(config.torch_device)

        self.node_id: torch.Tensor = torch.tensor([0]).to(config.torch_device)
        self.node_neighbor_id: torch.Tensor = torch.tensor([0]).to(config.torch_device)

        self.reward: torch.Tensor = torch.tensor([0], dtype=torch.float32).to(config.torch_device)

        self.all_score: torch.Tensor = torch.matmul(self.embedding_matrix, self.embedding_matrix.t()) + self.bias_vector
        self.node_embedding: torch.Tensor = torch.index_select(self.embedding_matrix, 0,
                                                               self.node_id).to(config.torch_device)
        self.node_neighbor_embedding: torch.Tensor = torch.index_select(self.embedding_matrix, 0,
                                                                        self.node_neighbor_id).to(config.torch_device)

        self.bias: torch.Tensor = torch.index_select(self.bias_vector, 0,
                                                     self.node_neighbor_id).to(config.torch_device)
        self.score: torch.Tensor = self.node_embedding * self.node_neighbor_embedding.sum(0)\
            .to(config.torch_device) + self.bias
        self.prob: torch.Tensor = utils.clip_by_tensor(torch.sigmoid(self.score), 1e-5, 1).to(config.torch_device)
