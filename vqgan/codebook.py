import torch

'''
    Codebook for VQGAN
'''
class CodeBook(torch.nn.Module):
    def __init__(self, num_vectors, latent_space_dim, beta):
        super(CodeBook, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.beta = beta

        self.book = torch.nn.Embedding(num_vectors, latent_space_dim)
        

    def forward(self, input):
        # flatten input to matrix of vectors
        z = input.permute(0, 2, 3, 1).contiguous()
        z = z.view(-1, self.latent_space_dim)

        # distances to codebook vectos
        d = torch.cdist(z, self.book.weight)

        # find closest vecs
        idx = torch.argmin(d, dim=1)
        z_q = self.book(idx).view(input.shape)

        # loss
        loss = torch.mean((z_q.detach() - input)**2) + self.beta * torch.mean((z_q - input.detach())**2)

        return z_q, idx, loss