from torch.autograd import Variable
import torch
from models.model_nn import Generator
import os
import random
from helpers import latent_space_vars


model_file = 'results/checkpoints/example-3936000.model'  # Vanilla model
hidden_size = 100
batch_size = 1

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

# Making alterations
n_alterations = 4
rand_indices = random.sample(range(100), n_alterations)
alterations = [-2.0, -1.0, 0.0, 1.0, 2.0]
z_init_bias = Variable(torch.rand(1, hidden_size))
z_batch = z_init_bias.clone().repeat(n_alterations*len(alterations), 1)
for i in range(n_alterations):
    for j in range(len(alterations)):
        z_idx = rand_indices[i]
        z_batch[(i*len(alterations))+j, z_idx] = z_batch[(i*len(alterations))+j, z_idx] + alterations[j]

latent_space_vars(batch=gen.forward(z_batch.view(-1, hidden_size, 1, 1)), directory=gen_images_dir,
                  indices=rand_indices, alterations=alterations, model_name='GAN')
