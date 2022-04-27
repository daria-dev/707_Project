import torch
import torch.nn as nn
import numpy as np

'''
    Bidirectional Transformer
'''

def weights_init(m):
    nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)

# https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_

class MultiHeadAttention(nn.Module):
    """
    # https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
    """
    def __init__(self, dim=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.Q = nn.Linear(dim, 16 * heads)
        self.K = nn.Linear(dim, 16 * heads)
        self.V = nn.Linear(dim, 16 * heads)

        self.heads = heads
        self.dim = dim

    def forward(self, q, k, v, mask):
        residual, batch_size = q, q.size(0)

        q_s = self.Q(q).view(batch_size, -1, self.heads, 16).transpose(1,2)
        k_s = self.K(k).view(batch_size, -1, self.heads, 16).transpose(1,2)
        v_s = self.V(v).view(batch_size, -1, self.heads, 16).transpose(1,2)

        attn_mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)

        #context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(16)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v_s)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * 16)
        output = nn.Linear(self.heads * 16, self.dim)(context)

        return nn.LayerNorm(self.dim)(output + residual), attn

class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim=768, hidden_dim=3072):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim)
        self.LayerNorm2 = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.GELU()
        ])

    def forward(self, x, mask):
        out, attn = self.MultiHeadAttention(x, x, x, mask)
        x = x.add(out)
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)
        return x

# https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
class BidirectionalTransformer(nn.Module):
    def __init__(self, N=24, dim=768, codebook_size=1024):
        super(BidirectionalTransformer, self).__init__()

        self.tok_emb = nn.Embedding(codebook_size, dim)
        self.pos_emb = nn.Embedding(8192, dim)
        self.norm = nn.LayerNorm(dim)

        self.EncoderLayers = nn.ModuleList([Encoder(dim) for _ in range(N)])
        self.Token_Prediction = nn.Linear(in_features=dim, out_features=codebook_size)
        #self.apply(weights_init)

        self.sos_token = 0
        self.num_image_tokens = 256
        self.mask_token_id = 1024
        self.choice_temperature = 4.5

        #self.activ1 = nn.Tanh()
        self.activ2 = nn.GELU()
        #self.fc = nn.Linear(dim, dim)
        #self.classifier = nn.Linear(dim, 2)
        self.linear = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        # decoder is shared with embedding layer
        embed_weight = self.tok_emb.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, x):
        # get seq embedding
        token_embeddings = self.tok_emb(x)

        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        position_embeddings = self.pos_emb(pos)

        embed = token_embeddings + position_embeddings
        embed = self.norm(embed)
        mask = get_attn_pad_mask(x, x)

        # pass through encoder
        for enc_layer in self.EncoderLayers:
            embed = enc_layer(embed, mask)

        # h_pooled = self.activ1(self.fc(embed[:, 0]))
        # logits_clsf = self.classifier(h_pooled)
        # masked_pos = masked_pos[:, :, None].expand(-1, -1, embed.size(-1))

        # h_masked = torch.gather(embed, 1, masked_pos)
        # h_masked = self.norm(self.activ2(self.linear(h_masked)))
        # logits_lm = self.decoder(h_masked) + self.decoder_bias
        emb = self.norm2(self.activ2(self.linear(embed)))
        token_pred = self.decoder(emb) + self.decoder_bias


        #tkn_prd = self.Token_Prediction(embed)
        return token_pred

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def gen_image(self, num=1, T=11):
        # start with blank tokens
        blank = torch.ones((num, self.num_image_tokens)) * (self.mask_token_id - 1)
        blank = blank.to(torch.int64)

        sos_tokens = torch.ones(blank.shape[0], 1, dtype=torch.long) * self.sos_token
        gen = torch.cat((sos_tokens, blank), dim=1)

        gamma = self.gamma_func()

        for t in range(T):
            # compute probabilities
            pred = self.forward(gen)
            pred_probs = torch.nn.functional.softmax(pred, dim=-1)

            # sample
            sampled = torch.distributions.categorical.Categorical(probs=pred_probs).sample()
            sampled_probs = torch.take(pred_probs, sampled)

            confidence_scores = torch.ones(gen.shape)
            masked = (gen == (self.mask_token_id - 1))
            confidence_scores = torch.where(masked, sampled_probs, confidence_scores)

            # mask schedule
            r = (t + 1.) / T
            mask_schedule = (int) (np.floor(gamma(r) * self.num_image_tokens))

            # select next tokens
            confidence_scores = torch.log(confidence_scores) + self.choice_temperature * (1.0 - r) * torch.distributions.gumbel.Gumbel(0, 1).sample(confidence_scores.shape)
            sorted_confidence, _ = torch.sort(confidence_scores, dim=-1)
            cut_off = sorted_confidence[:, mask_schedule]
            masked = (confidence_scores < cut_off)

            gen = torch.where(masked, self.mask_token_id - 1, sampled)

        return gen[:,1:]

