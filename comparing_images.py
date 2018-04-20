from torch.autograd import Variable
import torch
from models.model_wgan import Generator
from helpers import save_image_sample
import os


gan_model_file = 'results/checkpoints/example-2208000.model'  # Vanilla model
wgan_model_file = 'results/wgan_checkpoints/example-6112000.model'  # WGAN model
hidden_size = 100

# results save folder
gen_images_dir = 'assignment_results/generated_images'
if not os.path.isdir('assignment_results'):
    os.mkdir('assignment_results')
if not os.path.isdir(gen_images_dir):
    os.mkdir(gen_images_dir)

# Fixed to see how both models react
fixed_noise = Variable(torch.rand(16, hidden_size))

# WGAN model
from_before = torch.load(wgan_model_file, map_location=lambda storage, loc: storage)
wgan_gen = Generator(hidden_dim=hidden_size, leaky=0.2)
wgan_gen.load_state_dict(from_before['gen_state_dict'])

# Vanilla GAN model
from models.model_nn import Generator
from_before = torch.load(gan_model_file, map_location=lambda storage, loc: storage)
gan_gen = Generator(hidden_dim=hidden_size, leaky=0.2)
gan_gen.load_state_dict(from_before['gen_state_dict'])

# sample images for inspection
save_image_sample(batch=wgan_gen.forward(fixed_noise.view(-1, hidden_size, 1, 1)),
                  cuda=True, total_examples=0, directory=gen_images_dir, model_name='WGAN')

save_image_sample(batch=gan_gen.forward(fixed_noise.view(-1, hidden_size, 1, 1)),
                  cuda=True, total_examples=0, directory=gen_images_dir, model_name='GAN')
print("Saved images!")
