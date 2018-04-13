# stdlib imports
import argparse
import os

# thirdparty imports
from torch.utils.data import DataLoader
from torchvision import datasets

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

# from project
from model import Generator, Discriminator
from helpers import *


def main(train_set, learning_rate, n_epochs, beta_0, beta_1, batch_size, num_workers, hidden_size,
         model_file, cuda, display_result_every, checkpoint_interval, seed):

    #  make data between -1 and 1
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), train_set),
                                         transform=data_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers,
                                  drop_last=True)

    # initialize model
    if model_file != None:
        try:
            from_before = torch.load(model_file)
            total_examples = from_before['total_examples']
            gen_losses = from_before['gen_losses']
            disc_losses = from_before['disc_losses']
            gen_state_dict = from_before['gen_state_dict']
            disc_state_dict = from_before['disc_state_dict']

            # load generator and discriminator
            gen = Generator(hidden_dim=hidden_size)
            disc = Discriminator(leaky=0.2)
            disc.load_state_dict(disc_state_dict)
            gen.load_state_dict(gen_state_dict)
            print('model loaded successfully!')

        except:
            print('could not load model! creating new model...')
            model_file = None

    if model_file == None:
        print('creating new model...')
        gen = Generator(hidden_dim=hidden_size)
        disc = Discriminator(leaky=0.2)

        gen.weight_init(mean=0, std=0.02)
        disc.weight_init(mean=0, std=0.02)

        total_examples = 0
        disc_losses = []
        gen_losses = []

    if cuda:
        gen.cuda()
        disc.cuda()

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    gen_optimizer = optim.Adam(gen.parameters(), lr=learning_rate, betas=(beta_0, beta_1))
    disc_optimizer = optim.Adam(disc.parameters(), lr=learning_rate, betas=(beta_0, beta_1))

    # results save folder
    if not os.path.isdir('results/generated_images'):
        os.mkdir('results/generated_images')
    if not os.path.isdir('results/training_summaries'):
        os.mkdir('results/training_summaries')
    if not os.path.isdir('models'):
        os.mkdir('models')

    np.random.seed(seed)  # reset training seed to ensure that batches remain the same between runs!

    try:
        for epoch in range(n_epochs):
            disc_losses_epoch = []
            gen_losses_epoch = []
            for idx, (true_batch, _) in enumerate(train_dataloader):
                disc.zero_grad()

                #  Sample  minibatch  of examples from data generating distribution
                if cuda:
                    true_batch = Variable(true_batch.cuda())
                    true_target = Variable(torch.ones(batch_size).cuda())
                else:
                    true_batch = Variable(true_batch)
                    true_target = Variable(torch.ones(batch_size))

                #  train discriminator on true data
                true_disc_result = disc.forward(true_batch)
                disc_train_loss_true = BCE_loss(true_disc_result.squeeze(), true_target)
                disc_train_loss_true.backward()

                #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
                if cuda:
                    z = Variable(torch.zeros(batch_size, hidden_size).cuda())
                    fake_target = Variable(torch.zeros(batch_size).cuda())
                else:
                    z = Variable(torch.zeros(batch_size, hidden_size))
                    fake_target = Variable(torch.zeros(batch_size))

                #  train discriminator on fake data
                fake_batch = gen.forward(z.view(-1, hidden_size, 1, 1))
                fake_disc_result = disc.forward(fake_batch.detach())  # detach so gradients not computed for generator
                disc_train_loss_false = BCE_loss(fake_disc_result.squeeze(), fake_target)
                disc_train_loss_false.backward()

                #  compute performance statistics
                disc_train_loss = disc_train_loss_true + disc_train_loss_false
                disc_optimizer.step()
                disc_losses_epoch.append(disc_train_loss.data[0])

                disc_fake_accuracy = 1 - fake_disc_result.mean().data[0]
                disc_true_accuracy = true_disc_result.mean().data[0]

                #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
                if cuda:
                    z = Variable(torch.zeros(batch_size, hidden_size).cuda())
                    true_target = Variable(torch.ones(batch_size).cuda())
                else:
                    true_target = Variable(torch.ones(batch_size))
                    z = Variable(torch.zeros(batch_size, hidden_size))

                # train generator
                gen.zero_grad()
                fake_batch = gen.forward(z.view(-1, hidden_size, 1, 1))
                disc_result = disc.forward(fake_batch)
                gen_train_loss = BCE_loss(disc_result.squeeze(), true_target)

                gen_train_loss.backward()
                gen_optimizer.step()
                gen_losses_epoch.append(gen_train_loss.data[0])

                if (display_result_every != 0) and (total_examples % display_result_every == 0):
                    print('epoch {}: step {}/{} disc true acc: {:.4f} disc fake acc: {:.4f} '
                          'disc loss: {:.4f}, gen loss: {:.4f}'
                          .format(epoch+1, idx+1, len(train_dataloader), disc_true_accuracy, disc_fake_accuracy,
                                  disc_train_loss.data[0], gen_train_loss.data[0]))

                # Checkpoint model
                total_examples += batch_size
                if (checkpoint_interval != 0) and (total_examples % checkpoint_interval == 0):

                    disc_losses.extend(disc_losses_epoch)
                    gen_losses.extend(gen_losses_epoch)
                    save_checkpoint(total_examples, disc, gen, gen_losses, disc_losses)
                    print("Checkpoint saved!")

                    #  sample images for inspection
                    save_image_sample(batch=fake_batch, save_num=5, cuda=cuda, total_examples=total_examples)
                    print("Saved images!")


            print('epoch {}/{} disc loss: {:.4f}, gen loss: {:.4f}'
                  .format(epoch+1, n_epochs, np.array(disc_losses_epoch).mean(), np.array(gen_losses_epoch).mean()))

            disc_losses.extend(disc_losses_epoch)
            gen_losses.extend(gen_losses_epoch)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save_checkpoint(total_examples, disc, gen, gen_losses, disc_losses)
        print("Checkpoint saved!")

        #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
        if cuda:
            z = Variable(torch.zeros(batch_size, hidden_size).cuda())
        else:
            z = Variable(torch.zeros(batch_size, hidden_size))

        fake_batch = gen.forward(z.view(-1, hidden_size, 1, 1))

        # sample images for inspection
        save_image_sample(batch=fake_batch, save_num=5, cuda=cuda, total_examples=total_examples)
        print("Saved images!")


if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, default='data/resized_celebA')
    argparser.add_argument('--learning_rate', type=float, default=0.0002)
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--beta_0', type=float, default=0.5)
    argparser.add_argument('--beta_1', type=float, default=0.999)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--model_file', type=str, default=None)
    argparser.add_argument('--cuda', action='store_true', default=True)
    argparser.add_argument('--display_result_every', type=int, default=640)   # 640
    argparser.add_argument('--checkpoint_interval', type=int, default=32000)  # 32000
    argparser.add_argument('--seed', type=int, default=1024)
    args = argparser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    main(train_set=args.train_set, learning_rate=args.learning_rate, n_epochs=args.n_epochs,
         beta_0=args.beta_0, beta_1=args.beta_1, batch_size=args.batch_size, num_workers=args.num_workers,
         hidden_size=args.hidden_size, model_file=args.model_file, cuda=args.cuda,
         display_result_every=args.display_result_every,
         checkpoint_interval=args.checkpoint_interval, seed=args.seed)
