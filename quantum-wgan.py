# Library imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class GetDataset(Dataset):
    """Pytorch loader of transactions dataset"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transaction = self.df.iloc[idx, :-1]

        # Return image and label
        return transaction

    
# Loading the data from file
batch_size = 64

dataset = GetDataset(csv_file="creditcard.csv")
dataset.df.drop('Time', inplace=True, axis=1)

# selection only good (i.e. non-fraudulent data) for WGAN training
good = dataset.df[dataset.df['Class'] == 0]

dataset.df = pd.concat([good])

for col in dataset.df.columns:
    dataset.df[col] = (dataset.df[col] - dataset.df[col].min()) / (dataset.df[col].max() - dataset.df[col].min())

dataloader = torch.utils.data.DataLoader(
    torch.tensor(dataset).float(), batch_size=batch_size, shuffle=True
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        

class Discriminator(nn.Module):
    """
    Discriminator with fully connected layers
    """
    
    def __init__(self, n_features):
        """
        Args:
            n_features (int): Number of features of input data.
        """
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Linear(n_features, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.apply(init_weights)
                
    def forward(self, x):
        return self.discriminator(x)


def discriminator_loss(disc_fake_pred, disc_real_pred, gp, g_lambda):
    """
    Calculates the discrimination loss
    Args:
        disc_fake_pred (Tensor): Prediction of discriminator based on generated data.
        disc_real_pred (Tensor): Prediction of discriminator based on real data.
        gp (Tensor): Value of gradient penalty.
    """

    discriminator_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + g_lambda * gp
    return discriminator_loss

def get_gen_loss(disc_fake_pred):
    """
    Calculates the generator loss.
    Args:
        disc_fake_pred (Tensor): Prediction of discriminator based on generated data.
    """
    gen_loss = -1. * torch.mean(disc_fake_pred)
    return gen_loss

def gradient_penalty(disc, real_data, fake_data):
    """
    Calculates the gradient penalty in WGAN network.
    Args:
        disc (function): Discriminator used in network training.
        real_data (Tensor): Real data from training data
        fake_data (Tensor): Data obtained from the generator.
    """

    batch_size = real_data.size(0)
    eps = torch.rand(batch_size, 1).to(real_data.device)
    eps = eps.expand_as(real_data)
    interpolation = eps * real_data + (1 - eps) * fake_data
    
    interp_logits = disc(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    gradients = torch.autograd.grad(
        inputs=interpolation,
        outputs=interp_logits,
        grad_outputs=grad_outputs, 
        create_graph=True,
        retain_graph=True,
        
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    
    return torch.mean((grad_norm - 1) ** 2)

# define parameters of quantum circuit
n_qubits = 8
n_layers = 3

# Choice of quantum simulator
dev = qml.device("lightning.qubit", wires=n_qubits)
# Run simulation on CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(latents, theta, n_layers, rotation='Z'):
    """
    Quantum circuit used as a input part of generator.
    Args:
        latents (Parameter): Tensor of latent parameters used in first layer of circuit
        theta (Parameter): Parameters optimized during training of WGAN network
        n_layers (int): Number of layers in quantum circuit
        rotation (str): Rotation gate used in quantum circuit (default: 'Z')
        """
    
    theta = theta.reshape(n_layers, n_qubits)
    
    rot_gates = {'X': qml.RX, 'Y': qml.RY, 'Z': qml.RZ}
    rot_gate = rot_gates[rotation]
    
    # Initialize latent vectors
    for i in range(n_qubits):
        qml.RX(latents[i], wires=i)
    
    # Repeated layer
    for n in range(n_layers):
        for q in range(n_qubits):
            rot_gate(theta[n][q], wires=q)
        
        for q in range(1, n_qubits, 2):
            qml.CNOT(wires=[q - 1, q])
        for q in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[q, q + 1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumGenerator(nn.Module):
    """
    Quantum Generator with first part replaced with quantum circuit
    """
    
    def __init__(self, latents, n_features, n_layers, rotation='Z'):
        """
        Args:
            latents: Tensor of latent variables as input to the Generator.
            n_features (int): Number of features of input data.
            n_layers (int): Number of layers in quantum_circuit
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.latents = latents
        self.rotation = rotation
        self.q_params = nn.Parameter(torch.rand(n_layers * n_qubits), requires_grad=True)
        self.l_params = nn.Parameter(torch.rand(n_qubits), requires_grad=False)
        
        self.generator = nn.Sequential(
            nn.Linear(n_qubits, n_features),
            nn.Sigmoid()
        )
    
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.apply(init_weights)

    def forward(self, x):
        h1 = torch.as_tensor(self.latents)

        for i in range(batch_size):
            h1[i] = quantum_circuit(self.l_params, self.q_params, self.n_layers, self.rotation)
        h1.reshape(batch_size, self.latents.size(1))
        
        output = self.generator(h1)
        
        return output
    
    
lr = 0.0002        # learning rate for the Generator
beta_1 = 0.5       # optimizer coefficient for computing running average of gradient
beta_2 = 0.999     # optimizer coefficient for computing running average of gradient squared
adam_eps = 1e-7    # coefficient for improving numerical stability
num_iter = 1500    # number of training iterations

n_disc = 5         # number of discriminator training steps per one 

discriminator = Discriminator(n_features=29).to(device)

opt_disc = optim.Adam(discriminator.parameters(), lr=lr, eps=adam_eps, betas=(beta_1, beta_2))

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

g_lambda = 10

results = []

gen_losses = torch.zeros(num_iter)
disc_losses = torch.zeros(num_iter)

for it in range(num_iter):
    for i, data in enumerate(dataloader):
        for k in range(n_disc):
            for p in discriminator.parameters():
                    p.requires_grad = True
            
            real_data = data.to(device)

            theta = -2* np.pi * torch.rand(batch_size, n_qubits*n_layers, device=device, requires_grad=True) + np.pi
            latents = -2* np.pi * torch.rand(batch_size, n_qubits, device=device, requires_grad=False) + np.pi
            
            generator = QuantumGenerator(latents, n_features=29, n_layers=n_layers).to(device)
            opt_gen = optim.Adam(generator.parameters(), lr=0.01, eps=adam_eps, betas=(beta_1, beta_2))
            
            eps = torch.rand(batch_size, len(real_data[0]), device=device, requires_grad=True)
            
            fake_data = generator(theta)
            fake_data_hat = real_data + eps * (fake_data - real_data)

            # training the discriminator
            discriminator.zero_grad()
            
            disc_real = discriminator(real_data.float()).view(-1).to(device)
            disc_fake = discriminator(fake_data_hat.detach().float()).view(-1).to(device)
            
            gp = gradient_penalty(discriminator, real_data, fake_data)
            
            disc_loss = discriminator_loss(disc_fake, disc_real, gp, g_lambda)
            disc_losses[i] = disc_loss
            disc_loss.backward()
            opt_disc.step()
            
        
        # training the generator
        for p in discriminator.parameters():
                p.requires_grad = False # to avoid computation
        
        generator.zero_grad()
        discr_fake = discriminator(fake_data).view(-1)
        err_gen = get_gen_loss(discr_fake)
        gen_losses[i] = err_gen
        
        err_gen.backward()
        opt_gen.step()
        
    # show loss value
    if it % 10 == 0:
        print(f"Iteration: {it}, Discriminator loss: {disc_loss:0.3f}, Generator Loss: {err_gen:0.3f}")