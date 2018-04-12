import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import os


def save_image_sample(batch, save_num, cuda, total_examples):
    if not os.path.isdir('results/generated_images/{}'.format(total_examples)):
        os.mkdir('results/generated_images/{}'.format(total_examples))
    to_save = np.random.randint(batch.shape[0], size=save_num)
    for ix, k in enumerate(to_save):
        if cuda:
            plt.imshow(transforms.ToPILImage()(batch[k].data.cpu()))
        else:
            plt.imshow(transforms.ToPILImage()(batch[k]))

        plt.axis('off')
        plt.savefig('results/generated_images/{}/sample-{}'.format(total_examples, ix))


def save_checkpoint(total_examples, disc, gen, gen_losses, disc_losses):
    basename = "models/example-{}".format(total_examples)
    model_fname = basename + ".model"
    state = {
                'total_examples': total_examples,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'gen_losses': gen_losses,
                'disc_losses': disc_losses
            }
    torch.save(state, model_fname)
