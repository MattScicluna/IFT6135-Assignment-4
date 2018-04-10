# stdlib imports
import argparse
import os

# thirdparty imports
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np

# from project
from model import Generator

SEED = 1024


def see_image(tensor):
    assert isinstance(tensor, torch.FloatTensor)
    plt.imshow(transforms.ToPILImage()(tensor))


def main(train_set, learning_rate, n_epochs, batch_size, num_workers, hidden_size, model_file, cuda):

    data_transform = transforms.Compose([transforms.ToTensor(), ])

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), train_set),
                                         transform=data_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True)

    # initialize model
    if model_file == None:
        gen = Generator(hidden_dim=hidden_size)

    np.random.seed(SEED)  # reset training seed to ensure that batches remain the same between runs!
    for batch, _ in train_dataloader:

        batch = Variable(batch)
        #see_image(batch[0])
        if cuda:
            batch = batch.cuda()


if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, default='data/resized_celebA')
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--batch_size', type=int, default=25)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--model_file', type=str, default='None')
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    main(train_set=args.train_set, learning_rate=args.learning_rate, n_epochs=args.n_epochs,
         batch_size=args.batch_size, num_workers=args.num_workers, hidden_size=args.hidden_size,
         model_file=args.model_file, cuda=args.cuda)
