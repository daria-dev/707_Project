import torch
from transformer import BidirectionalTransformer
from vqgan import VQGAN
import argparse
from util import savefig_1

def main(args):
    # load models
    vqgan = VQGAN(args)
    checkpoint = torch.load(args.vqgan_chkpt)
    vqgan.load_state_dict(checkpoint)
    vqgan.eval()

    bert = BidirectionalTransformer()
    bert_chpt = torch.load(args.bert_chkpt)
    bert.load_state_dict(bert_chpt)
    bert.eval()

    for i in range(args.num_images):
        tokens = bert.gen_image()
        pred_flat = tokens.view(-1)

        # get codebook tokens
        tokens_img = torch.index_select(vqgan.codebook.embedding.weight,
                                                    dim=0, index=pred_flat)

        tokens_img = tokens_img.view((1, 16, 16, 256))
        tokens_img = tokens_img.permute(0, 3, 1, 2).contiguous()
                    
        img = vqgan.decode(tokens_img)
        savefig_1(img, "generated_" + str(i))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent_space_dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num_vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset', type=str, default='', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=50, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--data_preprocess', type=str, default='basic', help='Method for data pre-processing')
    parser.add_argument('--ext', type=str, default='*.jpg', help='Extension of data images')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=50)
    parser.add_argument('--vqgan_chkpt', type=str, default='vqgan_50', help='Path to pre-traiend vqgan model')
    parser.add_argument('--bert_chkpt', type=str, default='transformer_50', help='Path to pre-traiend vqgan model')
    parser.add_argument('--num_images', type=int, default=3)

    args = parser.parse_args()
    main(args)