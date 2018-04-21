import argparse
import sys
import os
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.inception import inception_v3
import numpy as np
from tqdm import *
from scipy.stats import entropy

from models import model_nn, model, model_bilinear, model_wgan


# TODO: adapt for transpose conv.

def inception_score(generator, nsamples=50000, batch_size=128, splits=10, resize=False, cuda=True):
    """
    Compute Inception Score given by \exp(\mathbb{E}_x [KL(p(y|x) || p(y)])
    Args:
        generator - (hopefully) trained GAN generator.
        nsamples - Number of samples to evaluate the IS on.
        batch_size - batch size for the generated images Dataloader.
        splits - number of splits to average the results.
        resize - True if images are smaller than 299x299.
        cuda - True if want to use GPU.
    """

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    inception_predictions = np.zeros((nsamples, 1000))

    # Load pretrained Inception model
    inception_net = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_net.eval();

    # Inception uses 299x299 inputs, so we might have to upsample
    def predict(x):
        if resize:
            x = F.upsample(x, size=(299, 299), mode='bilinear').type(dtype)
        x = inception_net(x)
        return F.softmax(x).data.cpu().numpy()

    # Compute the predictions (i.e. p(y|x)) using the Inception model
    for idx in tqdm(range(nsamples // batch_size)):
        # Generate random vectors to feed the generator
        z = Variable(torch.randn(args.batch_size, args.hidden_size).type(dtype))
        generated_imgs = generator(z)

        generated_imgs = generated_imgs.type(dtype)
        inception_predictions[idx * batch_size: idx * batch_size + batch_size] = predict(generated_imgs)

    # Compute the average KL, i.e. E_x[KL(p(y|x) || p(y))] and exponentiate it
    score_per_split = []
    for s in range(splits):
        split_scores = []
        split_preds = inception_predictions[s * (nsamples // splits): (s + 1) * (nsamples // splits), :]
        split_py = np.mean(split_preds, axis=0)

        for i in range(split_preds.shape[0]):
            p_y_given_xi = split_preds[i, :]
            split_scores.append(entropy(p_y_given_xi, split_py))
        score_per_split.append(np.exp(np.mean(split_scores)))

    # Return the mean over the splits
    return np.mean(score_per_split), np.std(score_per_split)


def mode_score(generator, train_set, nsamples_fake=50000, nsamples_real=50000,
               batch_size=128, splits=10, resize=False, cuda=True):
    """
    Compute Mode Score, given by \exp(\mathbb{E}_x [KL(p(y|x) || p(y)] - KL(p(y) || p(y*))
    where p(y*) = \int_x p(y|x) dP_{real}
    Args:
        generator - (hopefully) trained GAN generator.
        nsamples_fake - Number of samples of fake images to evaluate the MS on.
        nsamples_real - Number of samples of real images to evaluate the MS on.
        batch_size - batch size for the generated images Dataloader.
        splits - number of splits to average the results.
        resize - True if images are smaller than 299x299.
        cuda - True if want to use GPU.
    """

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    fake_predictions = np.zeros((nsamples_fake, 1000))
    real_predictions = np.zeros((nsamples_real, 1000))

    # Make real images between -1 and 1
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), train_set),
                                         transform=data_transform)
    train_dataset.imgs = train_dataset.imgs[:nsamples_real]

    # Dataloader of real images
    dataloader_real = DataLoader(train_dataset, batch_size=batch_size)

    # Load pretrained Inception model
    inception_net = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_net.eval();

    # Inception uses 299x299 inputs, so we might have to upsample
    def predict(x):
        if resize:
            x = F.upsample(x, size=(299, 299), mode='bilinear').type(dtype)
        x = inception_net(x)
        return F.softmax(x).data.cpu().numpy()

    # Compute the predictions (i.e. p(y|x)) using the Inception model
    for idx in tqdm(range(nsamples_fake // batch_size)):
        # Generate random vectors to feed the generator
        z = Variable(torch.randn(args.batch_size, args.hidden_size).type(dtype))
        generated_imgs = generator(z)
        generated_imgs = generated_imgs.type(dtype)

        fake_predictions[idx * batch_size: idx * batch_size + batch_size] = predict(generated_imgs)

    # Compute p(y|x) using samples from the real data
    for idx, x in tqdm(enumerate(dataloader_real, 0)):
        x = x[0].type(dtype)
        x_var = Variable(x)
        batch_size_ = x.size()[0]

        real_predictions[idx * batch_size: idx * batch_size + batch_size_] = predict(x_var)

    # Compute the average KL, i.e. E_x[KL(p(y|x) || p(y))] and KL(p(y) || p(y*))
    score_per_split = []
    for s in range(splits):
        split_scores = []
        split_preds_fake = fake_predictions[s * (nsamples_fake // splits): (s + 1) * (nsamples_fake // splits), :] #empty
        split_preds_real = real_predictions[s * (nsamples_fake // splits): (s + 1) * (nsamples_fake // splits), :]
        split_py_fake = np.mean(split_preds_fake, axis=0)   # p(y)  #nan
        split_py_real = np.mean(split_preds_real, axis=0)   # p(y*) #nan

        for i in range(split_preds_fake.shape[0]):
            p_y_given_xi_fake = split_preds_fake[i, :]
            split_scores.append(entropy(p_y_given_xi_fake, split_py_fake))
        score_per_split.append(np.exp(np.mean(split_scores) - entropy(split_py_fake, split_py_real)))
    # Return the mean over the splits
    return np.mean(score_per_split), np.std(score_per_split)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help="'nn', 'transpose', 'bilinear' or 'wgan' ")
    argparser.add_argument('--score', type=str, default='inception', help="'inception', 'mode' or 'both'")
    argparser.add_argument('--splits', type=int, default=10, help='Number of splits to compute the IS on.')
    argparser.add_argument('--nsamples', type=int, default=50000, help='Number of samples to evaluate the IS on.')
    argparser.add_argument('--nsamples_real', type=int, default=50000)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--hidden_size', type=int, default=100, help='Size of the random vector.')
    argparser.add_argument('--train_set', type=str, default='data/resized_celebA')
    argparser.add_argument('--cuda', action='store_true', default=False)
    args = argparser.parse_args()

    cuda = args.cuda and torch.cuda.is_available()

    if args.model == 'nn':
        G = model_nn.Generator(hidden_dim=args.hidden_size)
    elif args.model == 'transpose':
        G = model.Generator(hidden_dim=args.hidden_size)
    elif args.model == 'bilinear':
        G = model_bilinear.Generator(hidden_dim=args.hidden_size)
    elif args.model == 'wgan':
        G = model_wgan.Generator(hidden_dim=args.hidden_size)
    else:
        print("Wrong model name! Should be 'nn', 'transpose', 'bilinear' or 'wgan'")
        sys.exit()
    if cuda:
        G.cuda()

    # Inception Score
    if args.score == 'inception' or args.score == 'both':
        print('Computing Inception Score...')
        mean, std = inception_score(G, args.nsamples, args.batch_size, args.splits, resize=True, cuda=cuda)
        print('Inception Score = {} +- {}'.format(mean, std))
        np.savetxt('results/evaluation/inceptionScore_{}.txt'.format(args.model),
                   np.array([[mean], [std]]), delimiter=',', header='mean, std')

    # Inception Score
    if args.score == 'mode' or args.score == 'both':
        print('Computing Mode Score...')
        mean, std = mode_score(G, args.train_set, args.nsamples, args.nsamples_real,
                               args.batch_size, args.splits, resize=True, cuda=cuda)
        print('Mode Score = {} +- {}'.format(mean, std))
        np.savetxt('results/evaluation/modeScore_{}.txt'.format(args.model),
                   np.array([[mean], [std]]), delimiter=',', header='mean, std')