from torch.autograd import Variable
import torch
from models.model_nn import Generator
import os
import random
from helpers import latent_space_vars


model_file = 'results/checkpoints/example-2208000.model'  # Vanilla model
hidden_size = 100

# results save folder
gen_images_dir = 'assignment_results/latent_space_changes'
if not os.path.isdir('assignment_results'):
    os.mkdir('assignment_results')
if not os.path.isdir(gen_images_dir):
    os.mkdir(gen_images_dir)

# Loading model
from_before = torch.load(model_file, map_location=lambda storage, loc: storage)
gen = Generator(hidden_dim=hidden_size, leaky=0.2)
gen.load_state_dict(from_before['gen_state_dict'])
gen.eval()

# Making alterations
n_alterations = 4
rand_indices = random.sample(range(100), n_alterations)
alterations = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]

z_init_bias = Variable(torch.rand(1, hidden_size))
z_batch = Variable(torch.zeros([n_alterations*len(alterations), hidden_size]))
idxx = 0
for i in range(n_alterations):
    z_idx = rand_indices[i]
    for j in range(len(alterations)):
        z_batch[idxx, :] = z_init_bias
        z_batch[idxx, z_idx] = z_batch[idxx, z_idx] + alterations[j]
        idxx += 1

latent_space_vars(batch=gen.forward(z_batch.view(-1, hidden_size, 1, 1)), directory=gen_images_dir,
                  indices=rand_indices, alterations=alterations, model_name='GAN')
