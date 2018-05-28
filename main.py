import numpy as np
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from modules import VAE, VAE_NF
from data import binarized_mnist


def iterator(data, batch_size=32):
    for i in range(0, data.shape[0], batch_size):
        yield torch.from_numpy(data[i: i + batch_size]), None


BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 500
NUM_WORKERS = 4
LR = 2e-4
MODEL = 'VAE'  # VAE-NF | VAE

N_FLOWS = 30
Z_DIM = 40


n_steps = 0
writer = SummaryWriter()
dataset = binarized_mnist()

if MODEL == 'VAE-NF':
    model = VAE_NF(N_FLOWS, Z_DIM).cuda()
else:
    model = VAE(Z_DIM).cuda()

print(model)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)


def train():
    global n_steps
    train_loss = []
    model.train()
    train_loader = iterator(dataset['train'])

    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time()
        x = x.cuda().view(-1, 784)

        x_tilde, kl_div = model(x)
        loss_recons = F.binary_cross_entropy(x_tilde, x, size_average=False) / x.size(0)
        loss = loss_recons + kl_div

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append([loss_recons.item(), kl_div.item()])
        writer.add_scalar('loss/train/ELBO', loss.item(), n_steps)
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), n_steps)
        writer.add_scalar('loss/train/KL', kl_div.item(), n_steps)

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {:5.3f} ms/batch'.format(
                batch_idx * len(x), 50000,
                PRINT_INTERVAL * batch_idx / 50000,
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                1000 * (time.time() - start_time)
            ))

        n_steps += 1


def evaluate(split='valid'):
    global n_steps
    start_time = time.time()
    val_loss = []
    model.eval()
    eval_loader = iterator(dataset[split])

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(eval_loader):
            x = x.cuda().view(-1, 784)

            x_tilde, kl_div = model(x)
            loss_recons = F.binary_cross_entropy(x_tilde, x, size_average=False) / x.size(0)
            loss = loss_recons + kl_div

            val_loss.append(loss.item())
            writer.add_scalar('loss/{}/ELBO'.format(split), loss.item(), n_steps)
            writer.add_scalar('loss/{}/reconstruction'.format(split), loss_recons.item(), n_steps)
            writer.add_scalar('loss/{}/KL'.format(split), kl_div.item(), n_steps)

    print('\nEvaluation Completed ({})!\tLoss: {:5.4f} Time: {:5.3f} s'.format(
        split,
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_reconstructions():
    model.eval()
    x = torch.from_numpy(dataset['test'][:32])

    x = x[:32].cuda().view(-1, 784)
    x_tilde, _ = model(x)

    x_cat = torch.cat([x, x_tilde], 0).view(-1, 1, 28, 28)
    images = x_cat.cpu().data

    save_image(
        images,
        'samples/{}_reconstructions.png'.format(MODEL),
        nrow=8
    )


def generate_samples():
    model.eval()
    z = torch.randn(64, Z_DIM).cuda()
    x_tilde = model.decoder(z).view(-1, 1, 28, 28)
    images = x_tilde.cpu().data
    save_image(
        images,
        'samples/{}_samples.png'.format(MODEL),
        nrow=8
    )


BEST_LOSS = 99999
LAST_SAVED = -1
for epoch in range(1, N_EPOCHS):
    print("Epoch {}:".format(epoch))
    train()
    cur_loss = evaluate()

    if cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model!")
        torch.save(model.state_dict(), 'models/{}.pt'.format(MODEL))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_reconstructions()
    generate_samples()
