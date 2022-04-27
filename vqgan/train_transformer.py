import torch
import argparse
import numpy as np
import torch.nn.functional as F
from util import get_data_loader, savefig
from vqgan import VQGAN
from transformer import BidirectionalTransformer
from util import savefig

def main(args):
    # load images
    dataloader = get_data_loader(args.dataset, args)

    # load models
    vqgan = VQGAN(args)
    checkpoint = torch.load(args.vqgan_chkpt)
    vqgan.load_state_dict(checkpoint)
    vqgan.eval()

    transformer = BidirectionalTransformer()

    # configure optimizer
    optim = torch.optim.Adam(
        transformer.parameters(),
        lr=args.lr, eps=1e-08, betas=(args.beta1, args.beta2)
    )
    total_train_iters = args.epochs * len(dataloader)

    iters = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            real_images, labels = batch
            optim.zero_grad()

            # get codebook tokens
            latents, idx = vqgan.encode(real_images)
            idx = idx.view(real_images.shape[0], -1)

            r = np.random.uniform()
            mask = torch.bernoulli(r * torch.ones(idx.shape))

            masked_indices = (1024 -1) * torch.zeros_like(idx)
            y = mask * idx + (1 - mask) * masked_indices
            y = torch.tensor(y, dtype=torch.long)

            # add sos token to beginning
            sos_tokens = torch.ones(y.shape[0], 1, dtype=torch.long) * transformer.sos_token
            y = torch.cat((sos_tokens, y), dim=1)
            idx = torch.cat((sos_tokens, idx), dim=1)

            # pred
            pred = transformer(y)

            # step
            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), idx.reshape(-1))
            loss.backward()
            optim.step()

            if iters % 10 == 0:
                print('Iteration [{:4d}/{:4d}] | loss: {:6.4f}'.format(
                       iters, total_train_iters,  loss.item()))

            if iters % 50 == 0:
                with torch.no_grad():
                    # get token indecies
                    #tokens = transformer.sample_good()
                    #tokens_flat = tokens.view(-1)
                    pred_prob = torch.nn.functional.softmax(pred, dim=-1)
                    pred,_ = torch.max(pred_prob, dim=2)
                    pred = pred.to(torch.int64)

                    # replace maskd with predicted
                    masked = ~(y == idx)
                    pred = torch.where(masked, pred, y)[:,1:]
                    pred_flat = pred.contiguous().view(-1)

                    # get codebook tokens
                    tokens_img = torch.index_select(vqgan.codebook.embedding.weight,
                                                    dim=0, index=pred_flat)

                    tokens_img = tokens_img.view_as(latents)
                    tokens_img = tokens_img.permute(0, 3, 1, 2).contiguous()
                    
                    img = vqgan.decode(tokens_img)

                    fake = torch.clamp(img[0,:,:,:], 0, 1)
                    savefig(real_images[0,:,:,:], fake, iters)

            iters += 1

    torch.save(transformer.state_dict(), "transformer_"+ str(args.epochs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument('--dataset', type=str, default='', help='Path to data (default: /data)')
    parser.add_argument('--vqgan_chkpt', type=str, default='vqgan_50', help='Path to pre-traiend vqgan model')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--data_preprocess', type=str, default='basic', help='Method for data pre-processing')
    parser.add_argument('--ext', type=str, default='*.jpg', help='Extension of data images')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 6)')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--latent_space_dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--num_vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--lr', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')

    args = parser.parse_args()
    main(args)