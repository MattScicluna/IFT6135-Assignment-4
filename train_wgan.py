# stdlib imports
import argparse
import os

# thirdparty imports
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.model_wgan import Generator, Discriminator, wgan_Dloss, wgan_Gloss

# from project
from helpers import load_model_wgan, save_checkpoint, save_image_sample, save_learning_curve, save_learning_curve_epoch


def main(train_set, learning_rate, n_epochs, batch_size, num_workers, hidden_size, model_file,
         cuda, checkpoint_interval, seed, n_disc):

    #  make data between -1 and 1
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), train_set),
                                         transform=data_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True)

    # initialize model
    if model_file:
        try:
            total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
            disc_loss_per_epoch, prev_epoch, gen, disc = load_model_wgan(model_file, hidden_size)  # TODO: upsampling method?
            print('model loaded successfully!')

        except:
            print('could not load model! creating new model...')
            model_file = None

    if not model_file:
        print('creating new model...')

        gen = Generator(hidden_dim=hidden_size, leaky=0.2)
        disc = Discriminator(leaky=0.2)

        gen.weight_init(mean=0, std=0.02)
        disc.weight_init(mean=0, std=0.02)

        total_examples = 0
        disc_losses = []
        gen_losses = []
        disc_loss_per_epoch = []
        gen_loss_per_epoch = []
        prev_epoch = 0

        #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
        if cuda:
            fixed_noise = Variable(torch.randn(9, hidden_size).cuda())
        else:
            fixed_noise = Variable(torch.rand(9, hidden_size))

    if cuda:
        gen.cuda()
        disc.cuda()

    # Adam optimizer
    gen_optimizer = optim.RMSprop(gen.parameters(), lr=learning_rate, eps=1e-8)
    disc_optimizer = optim.RMSprop(disc.parameters(), lr=learning_rate, eps=1e-8)

    # results save folder
    gen_images_dir = 'results/wgan_generated_images'
    train_summaries_dir = 'results/wgan_training_summaries'
    checkpoint_dir = 'results/wgan_checkpoints'
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(gen_images_dir):
        os.mkdir(gen_images_dir)
    if not os.path.isdir(train_summaries_dir):
        os.mkdir(train_summaries_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    np.random.seed(seed)  # reset training seed to ensure that batches remain the same between runs!

    try:
        for epoch in range(prev_epoch, n_epochs):
            disc_losses_epoch = []
            gen_losses_epoch = []
            for idx, (true_batch, _) in enumerate(train_dataloader):
                disc.zero_grad()

                #  Sample  minibatch  of examples from data generating distribution
                if cuda:
                    true_batch = Variable(true_batch.cuda())
                else:
                    true_batch = Variable(true_batch)

                # discriminator on true data
                true_disc_output = disc.forward(true_batch)

                #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
                if cuda:
                    z = Variable(torch.randn(batch_size, hidden_size).cuda())
                else:
                    z = Variable(torch.randn(batch_size, hidden_size))

                # discriminator on fake data
                # fake_batch = gen.forward(z.view(-1, hidden_size, 1, 1))
                fake_batch = gen.forward(z)
                fake_disc_output = disc.forward(
                    fake_batch.detach())  # detach so gradients not computed for generator

                # Optimize with new loss function
                disc_loss = wgan_Dloss(true_disc_output, fake_disc_output)
                disc_loss.backward()
                disc_optimizer.step()

                # Weight clipping as done by WGAN
                for p in disc.parameters():
                    p.data.clamp_(-0.01, 0.01)

                #  Store losses
                disc_losses_epoch.append(disc_loss.data[0])

                # Train generator after the discriminator has been trained n_disc times
                if (idx+1) % n_disc == 0:
                    gen.zero_grad()

                    # Sample minibatch of m noise samples from noise prior p_g(z) and transform
                    if cuda:
                        z = Variable(torch.randn(batch_size, hidden_size).cuda())
                    else:
                        z = Variable(torch.rand(batch_size, hidden_size))

                    # train generator
                    fake_batch = gen.forward(z.view(-1, hidden_size, 1, 1))
                    fake_disc_output = disc.forward(fake_batch)

                    # Optimize generator
                    gen_loss = wgan_Gloss(fake_disc_output)
                    gen_loss.backward()
                    gen_optimizer.step()

                    # Store losses
                    gen_losses_epoch.append(gen_loss.data[0])

                if (idx != 0) and ((idx+1) % n_disc*4 == 0):
                    print('epoch {}: step {}/{} disc loss: {:.4f}, gen loss: {:.4f}'
                          .format(epoch + 1, idx + 1, len(train_dataloader), disc_loss.data[0], gen_loss.data[0]))

                # Checkpoint model
                total_examples += batch_size
                if (checkpoint_interval != 0) and (total_examples % checkpoint_interval == 0):
                    disc_losses.extend(disc_losses_epoch)
                    gen_losses.extend(gen_losses_epoch)
                    save_checkpoint(total_examples=total_examples, fixed_noise=fixed_noise, disc=disc, gen=gen,
                                    gen_losses=gen_losses, disc_losses=disc_losses,
                                    disc_loss_per_epoch=disc_loss_per_epoch,
                                    gen_loss_per_epoch=gen_loss_per_epoch, epoch=epoch, directory=checkpoint_dir)
                    print("Checkpoint saved!")

                    #  sample images for inspection
                    save_image_sample(batch=gen.forward(fixed_noise.view(-1, hidden_size, 1, 1)),
                                      cuda=cuda, total_examples=total_examples, directory=gen_images_dir)
                    print("Saved images!")

                    # save learning curves for inspection
                    save_learning_curve(gen_losses=gen_losses, disc_losses=disc_losses,
                                        total_examples=total_examples, directory=train_summaries_dir)
                    print("Saved learning curves!")

            disc_loss_per_epoch.append(np.average(disc_losses_epoch))
            gen_loss_per_epoch.append(np.average(gen_losses_epoch))

            # Save epoch learning curve
            save_learning_curve_epoch(gen_losses=gen_loss_per_epoch, disc_losses=disc_loss_per_epoch,
                                      total_epochs=epoch + 1, directory=train_summaries_dir)
            print("Saved learning curves!")

            print('epoch {}/{} disc loss: {:.4f}, gen loss: {:.4f}'
                  .format(epoch + 1, n_epochs, np.array(disc_losses_epoch).mean(), np.array(gen_losses_epoch).mean()))

            disc_losses.extend(disc_losses_epoch)
            gen_losses.extend(gen_losses_epoch)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save_checkpoint(total_examples=total_examples, fixed_noise=fixed_noise, disc=disc, gen=gen,
                        disc_loss_per_epoch=disc_loss_per_epoch,
                        gen_loss_per_epoch=gen_loss_per_epoch,
                        gen_losses=gen_losses, disc_losses=disc_losses, epoch=epoch, directory=checkpoint_dir)
        print("Checkpoint saved!")

        # sample images for inspection
        save_image_sample(batch=gen.forward(fixed_noise.view(-1, hidden_size, 1, 1)),
                          cuda=cuda, total_examples=total_examples, directory=gen_images_dir)
        print("Saved images!")

        # save learning curves for inspection
        save_learning_curve(gen_losses=gen_losses, disc_losses=disc_losses,
                            total_examples=total_examples, directory=train_summaries_dir)
        print("Saved learning curves!")


if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, default='data/resized_celebA')
    argparser.add_argument('--learning_rate', type=float, default=0.00005)
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--model_file', type=str, default=None)
    argparser.add_argument('--cuda', action='store_true', default=False)
    argparser.add_argument('--checkpoint_interval', type=int, default=32000)  # 32000
    argparser.add_argument('--seed', type=int, default=1024)
    argparser.add_argument('--n_disc', type=int, default=5)
    args = argparser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    print("Using GPU?", args.cuda)

    main(train_set=args.train_set,
         learning_rate=args.learning_rate,
         n_epochs=args.n_epochs,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         hidden_size=args.hidden_size,
         model_file=args.model_file,
         cuda=args.cuda,
         checkpoint_interval=args.checkpoint_interval,
         seed=args.seed,
         n_disc=args.n_disc)
