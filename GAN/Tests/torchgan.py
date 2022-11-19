import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Pixel Values scale to 0 ~ 1 -> -1 ~ 1
])

mnist = datasets.MNIST(root='data', download=True, transform=transform)

dataloader = DataLoader(mnist, batch_size=60, shuffle=True)


if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

leave_log = True

if leave_log:
    result_dir = 'GAN_generated_images'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh())

    def forward(self, inputs):
        return self.main(inputs).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid())

    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        return self.main(inputs)

G = Generator()
D = Discriminator()

if use_gpu:
    G.cuda()
    D.cuda()

# Binary Cross Entropy loss
criterion = nn.BCELoss()

# Adam optimizer
G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Visualization
def square_plot(data, path):
    if type(data) == list:
        data = np.concatenate(data)
    data = (data - data.min()) / (data.max() - data.min())

    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))
               + ((0, 0),) * (data.ndim - 3))
    data = np.pad(data, padding, mode='constant',
                  constant_values=1)

    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])


if leave_log:
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    generated_images = []

z_fixed = Variable(torch.randn(5 * 5, 100), volatile=True)
if use_gpu:
    z_fixed = z_fixed.cuda()

n_epoch = 200

for epoch in range(n_epoch):
    if leave_log:
        D_losses = []
        G_losses = []

    for real_data, _ in dataloader:
        batch_size = real_data.size(0)

        real_data = Variable(real_data)
        target_real = Variable(torch.ones(batch_size, 1))
        target_fake = Variable(torch.zeros(batch_size, 1))

        if use_gpu:
            real_data, target_real, target_fake = real_data.cuda(
            ), target_real.cuda(), target_fake.cuda()

        D_result_from_real = D(real_data)

        D_loss_real = criterion(D_result_from_real, target_real)

        z = Variable(torch.randn((batch_size, 100)))

        if use_gpu:
            z = z.cuda()

        fake_data = G(z)

        D_result_from_fake = D(fake_data)
        D_loss_fake = criterion(D_result_from_fake, target_fake)
        D_loss = D_loss_real + D_loss_fake
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        if leave_log:
            D_losses.append(D_loss.data[0])

        z = Variable(torch.randn((batch_size, 100)))

        if use_gpu:
            z = z.cuda()

        fake_data = G(z)
        D_result_from_fake = D(fake_data)
        G_loss = criterion(D_result_from_fake, target_real)
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if leave_log:
            G_losses.append(G_loss.data[0])
    if leave_log:
        true_positive_rate = (D_result_from_real > 0.5).float().mean().data[0]
        true_negative_rate = (D_result_from_fake < 0.5).float().mean().data[0]
        base_message = ("Epoch: {epoch:<3d} D Loss: {d_loss:<8.6} G Loss: {g_loss:<8.6} "
                        "True Positive Rate: {tpr:<5.1%} True Negative Rate: {tnr:<5.1%}"
                        )
        message = base_message.format(
            epoch=epoch,
            d_loss=sum(D_losses)/len(D_losses),
            g_loss=sum(G_losses)/len(G_losses),
            tpr=true_positive_rate,
            tnr=true_negative_rate
        )
        print(message)

    if leave_log:
        fake_data_fixed = G(z_fixed)
        image_path = result_dir + '/epoch{}.png'.format(epoch)
        square_plot(fake_data_fixed.view(
            25, 28, 28).cpu().data.numpy(), path=image_path)
        generated_images.append(image_path)

    if leave_log:
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

torch.save(G.state_dict(), "gan_generator.pkl")
torch.save(D.state_dict(), "gan_discriminator.pkl")
with open('gan_train_history.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

generated_image_array = [imageio.imread(
    generated_image) for generated_image in generated_images]
imageio.mimsave(result_dir + '/GAN_generation.gif',
                generated_image_array, fps=5)
