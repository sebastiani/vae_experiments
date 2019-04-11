import os
import torch
import torch.optim as optim
import torchvision.datasets as datasets
from src.models.GaussianVAE import GaussianMLP_VAE
from torchvision import transforms
import src.loss as losses


def train(cfg):
    loss_func = getattr(losses, cfg['loss'])

    root_dir = cfg['dataset']['path']
    download =  os.path.isdir(root_dir)  # check if data is already downloaded
    dataset = getattr(datasets, cfg['datasets']['name'])

    transform = transforms.ToTensor()
    train_dataset = dataset(root=root_dir, download=download, transforms=transform)
    train_loader = torch.util.data.DataLoader(train_dataset,
                                              batch_size=cfg['batch_size'],
                                              shuffle=True,
                                              num_workers=cfg['num_workers'])
    test_dataset = dataset(root_dir=root_dir, train=False, download=download, transforms=transform)
    test_loader = torch.util.data.DataLoader(test_dataset,
                                             batch_size=cfg['batch_size'],
                                             shuffle=True,
                                             num_workers=cfg['num_workers'])

    model = GaussianMLP_VAE(cfg)
    model = model.to(cfg['device'])

    optimizer = getattr(optim, cfg['optimizer']['type'])(model.parameters(), **cfg['optimizer']['args'])
    running_loss = 0.0
    for epoch in range(cfg['epochs']):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(cfg['device']), labels.to(cfg['device'])

            optimizer.zero_grad()

            output, z_mu, z_logvar = model(inputs)

            loss = loss_func(inputs, output, z_mu, z_logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print("[{}, {}] loss: {}".format(epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

            #Need to save checkpoints here...

    print("Finished Training")