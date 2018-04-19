import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from models.model import Generator, Discriminator  # TODO: other upsampling methods?


def save_image_sample(batch, cuda, total_examples):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   transforms.ToPILImage()
                                   ])

    f, axarr = plt.subplots(nrows=int(np.sqrt(len(batch))), ncols=int(np.sqrt(len(batch))))
    indx = 0
    for i in range(int(np.sqrt(len(batch)))):
        for j in range(int(np.sqrt(len(batch)))):
            if cuda:
                axarr[i, j].imshow(invTrans(batch[indx].data.cpu()))
                indx += 1
            else:
                axarr[i, j].imshow(invTrans(batch[indx]))
                indx += 1

            # Turn off tick labels
            axarr[i, j].axis('off')

    f.tight_layout()
    f.savefig('results/generated_images/gen_images_after_{}_examples'.format(total_examples))


def save_checkpoint(total_examples, disc, gen, gen_losses, disc_losses,
                    disc_loss_per_epoch, gen_loss_per_epoch, fixed_noise, directory):
    basename = directory+"/example-{}".format(total_examples)
    model_fname = basename + ".model"
    state = {
        'total_examples': total_examples,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'gen_losses': gen_losses,
        'disc_losses': disc_losses,
        'disc_loss_per_epoch': disc_loss_per_epoch,
        'gen_loss_per_epoch': gen_loss_per_epoch,
        'fixed_noise': fixed_noise
    }
    torch.save(state, model_fname)


def compute_model_stats(model):
    num_weights = 0
    for params in model.parameters():
        num_weights += params.numel()
    print('there are {} parameters'.format(num_weights))

    for params in model.parameters():
        print('avg weight value: {:.3f}'.format(params.mean().data[0]))


def load_model(model_file, hidden_size):
    from_before = torch.load(model_file)
    total_examples = from_before['total_examples']
    gen_losses = from_before['gen_losses']
    disc_losses = from_before['disc_losses']
    gen_loss_per_epoch = from_before['gen_loss_per_epoch']
    disc_loss_per_epoch = from_before['disc_loss_per_epoch']
    gen_state_dict = from_before['gen_state_dict']
    disc_state_dict = from_before['disc_state_dict']
    fixed_noise = from_before['fixed_noise']

    # load generator and discriminator
    gen = Generator(hidden_dim=hidden_size)
    disc = Discriminator(leaky=0.2)
    disc.load_state_dict(disc_state_dict)
    gen.load_state_dict(gen_state_dict)
    return total_examples, fixed_noise, gen_losses, disc_losses, \
           gen_loss_per_epoch, disc_loss_per_epoch, gen, disc


def save_learning_curve(gen_losses, disc_losses, total_examples):
    plt.figure()
    #plt.title('GAN Learning Curves')
    plt.plot(gen_losses, color='red', label='Generator')
    plt.plot(disc_losses, color='blue', label='Discriminator')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('results/training_summaries/learn_curves_after_{}_examples'.format(total_examples))


def save_learning_curve_epoch(gen_losses, disc_losses, total_epochs):
    plt.figure()
    #plt.title('GAN Learning Curves')
    plt.plot(gen_losses, color='red', label='Generator')
    plt.plot(disc_losses, color='blue', label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('results/training_summaries/learn_curves_after_{}_epochs'.format(total_epochs))