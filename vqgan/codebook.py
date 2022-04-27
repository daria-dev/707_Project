import torch
from torch.autograd import Function

'''
    Codebook for VQGAN
'''
# class CodeBook(torch.nn.Module):
#     def __init__(self, num_vectors, latent_space_dim, beta):
#         super(CodeBook, self).__init__()
#         self.latent_space_dim = latent_space_dim
#         self.beta = beta

#         self.book = torch.nn.Embedding(num_vectors, latent_space_dim)
        

#     def forward(self, input):
#         # flatten input to matrix of vectors
#         z = input.permute(0, 2, 3, 1).contiguous()
#         z = z.view(-1, self.latent_space_dim)

#         # distances to codebook vectos
#         with torch.no_grad():
#             d = torch.cdist(z, self.book.weight)

#             # find closest vecs
#             idx = torch.argmin(d, dim=1)

#         z_q = self.book(idx).view(input.shape)

#         # loss
#         loss = torch.mean((z_q.detach() - input)**2) + self.beta * torch.mean((z_q - input.detach())**2)

#         return z_q, idx, loss
class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')
        

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)
    
vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply


class VQEmbedding(torch.nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = torch.nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents, idx = vq_st(z_e_x_, self.embedding.weight)
        return latents, idx

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        loss = torch.mean((z_q_x.detach() - z_e_x)**2) + 0.25 * torch.mean((z_q_x - z_e_x.detach())**2)

        return z_q_x, z_q_x_bar, loss