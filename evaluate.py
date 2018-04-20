import argparse
import sys
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
import numpy as np
from tqdm import *
from scipy.stats import entropy

from models import model_nn, model, model_bilinear, model_wgan


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

    #num_imgs = len(generated_imgs)
    #inception_predictions = np.zeros((num_imgs, 1000))
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
        inception_predictions[idx*batch_size: idx*batch_size + batch_size] = predict(generated_imgs)

    # Compute the average KL, i.e. E_x[KL(p(y|x) || p(y))] and exponentiate it
    score_per_split = []
    for s in range(splits):
        split_scores = []
        split_preds = inception_predictions[s * (nsamples // splits): (s+1) * (nsamples // splits), :]
        split_py = np.mean(split_preds, axis=0)

        for i in range(split_preds.shape[0]):
            p_y_given_xi = split_preds[i, :]
            split_scores.append(entropy(p_y_given_xi, split_py))
        score_per_split.append(np.exp(np.mean(split_scores)))

    # Return the mean over the splits
    return np.mean(score_per_split), np.std(score_per_split)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help="'nn', 'transpose', 'bilinear' or 'wgan' ")
    argparser.add_argument('--splits', type=int, default=10, help='Number of splits to compute the IS on.')
    argparser.add_argument('--nsamples', type=int, default=50000, help='Number of samples to evaluate the IS on.')
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--hidden_size', type=int, default=100, help='Size of the random vector.')
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

    # Inception Score
    print('Computing Inception Score...')
    mean, std = inception_score(G, args.nsamples, args.batch_size, args.splits, resize=True, cuda=cuda)
    print('IS = {} +- {}'.format(mean, std))
    np.savetxt('results/evaluation/{}.txt'.format(args.model),
               np.array([[mean], [std]]), delimiter=',', header='mean, std')

