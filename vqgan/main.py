from vqgan import VQGAN
from util import get_data_loader, imshow, PerceptualLoss
from disc import Discriminator
import argparse
import matplotlib.pyplot as plt
import torch

'''
    GAN Loss Adaptive Weight
'''
def get_lambda(G, rec_loss, gan_loss, delta):
    last_layer = G.decoder.model[-1].weight
    gan_loss_grad = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
    rec_loss_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]

    return torch.norm(rec_loss_grad) / (torch.norm(gan_loss_grad) + delta)

'''
    Training Loop
'''
def main(args):
    # load images
    dataloader = get_data_loader(args.dataset, args)

    # interative MPL
    plt.ion()

    # preceptual loss
    PL = PerceptualLoss()

    # models
    G = VQGAN(args)
    D = Discriminator()

    # optimizers
    lr = args.learning_rate
    vq_opt = torch.optim.Adam(
        list(G.encoder.parameters()) +
        list(G.decoder.parameters()) +
        list(G.codebook.parameters()) +
        list(G.enc_conv.parameters()) +
        list(G.code_conv.parameters()),
        lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
    )

    d_opt = torch.optim.Adam(
        D.parameters(),
        lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
    )

    total_train_iters = args.epochs * len(dataloader)
    iteration = 1

    for i in range(args.epochs):
        for batch in dataloader:
            real_images, labels = batch

            # reconstruct images through vq
            reconst_images, _, _ = G(real_images)

            # discriminator results
            disc_real = D(real_images)
            disc_fake = D(reconst_images.detach())

            # train discriminator
            gan_loss = (disc_real - 1)**2 + (disc_fake)**2
            gan_loss = 0.5 * torch.mean(gan_loss)
            gan_loss_print = gan_loss

            d_opt.zero_grad()
            gan_loss.backward()
            d_opt.step()

            # train vq
            reconst_images, _, codebook_loss = G(real_images)
            disc_fake = D(reconst_images)

            # losses
            gan_loss = torch.mean((disc_fake - 1)**2)
            preceptual_loss = PL(real_images, reconst_images)
            gan_lambda = get_lambda(G, preceptual_loss, gan_loss, 1e-4)

            vq_loss = codebook_loss + preceptual_loss + gan_lambda * gan_loss

            # step
            vq_opt.zero_grad()
            vq_loss.backward()
            vq_opt.step()

            iteration += 1

            # print losses
            if iteration % args.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | GAN_loss: {:6.4f} | VQ_loss: {:6.4f}'.format(
                       iteration, total_train_iters, gan_loss_print.item(), vq_loss.item()))

            # show images
            if iteration % args.sample_every == 0:
                plt.figure()
                imshow(real_images[0,:,:,:], title="Real Image")

                plt.figure()
                imshow(reconst_images[0,:,:,:], title="Reconstructed Image")

    plt.ioff()
    plt.show()


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
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--data_preprocess', type=str, default='basic', help='Method for data pre-processing')
    parser.add_argument('--ext', type=str, default='*.jpg', help='Extension of data images')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)

    args = parser.parse_args()
    main(args)
