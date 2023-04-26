from src.GraphGAN.graph_gan import GraphGAN
# import torch

def main():
    model = GraphGAN()
    model.train()
    # t = torch.randn(1, 50) * 5
    # print("t is",t)
    # t[t < 1] = 1
    # print("new t is", t)

if __name__ == '__main__':
    main()
