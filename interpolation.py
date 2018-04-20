from torch.autograd import Variable
import torch
from models.model_nn import Generator
import os
from helpers import save_interpolation_sample


model_file = 'results/checkpoints/example-2208000.model'  # Vanilla model
hidden_size = 100

# results save folder
gen_images_dir = 'assignment_results/interpolations'
if not os.path.isdir('assignment_results'):
    os.mkdir('assignment_results')
if not os.path.isdir(gen_images_dir):
    os.mkdir(gen_images_dir)

# Loading model
from_before = torch.load(model_file, map_location=lambda storage, loc: storage)
gen = Generator(hidden_dim=hidden_size, leaky=0.2)
gen.load_state_dict(from_before['gen_state_dict'])
gen.eval()

# The weighting for interpolation
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# In the latent space Z
z_0 = Variable(torch.randn(1, hidden_size))
z_1 = Variable(torch.randn(1, hidden_size))
z_batch = Variable(torch.zeros([len(alphas), hidden_size]))
idxx = 0
for alpha in alphas:
    z_prime = alpha*z_0 + (1 - alpha)*z_1
    z_batch[idxx, :] = z_prime
    idxx += 1

save_interpolation_sample(batch=gen.forward(z_batch.view(-1, hidden_size, 1, 1)), directory=gen_images_dir,
                  alphas=alphas, model_name='GAN-Z')

# In the image space X using same z_0 and z_1 above
x_0 = gen.forward(z_0.view(-1, hidden_size, 1, 1))
x_1 = gen.forward(z_1.view(-1, hidden_size, 1, 1))

x_batch = Variable(torch.zeros([len(alphas), 3, 64, 64]))
idxx = 0
for alpha in alphas:
    x_prime = alpha*x_0 + (1 - alpha)*x_1
    x_batch[idxx, :] = x_prime
    idxx += 1

save_interpolation_sample(batch=x_batch, directory=gen_images_dir, alphas=alphas, model_name='GAN-X')
