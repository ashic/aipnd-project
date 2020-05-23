import argparse
import os 
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from collections import OrderedDict
from PIL import Image
import logging

def validation(model, validation_loader, criterion, device):
    model.to (device)
    
    valid_loss = 0
    accuracy = 0

    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


def train(device, model, optimiser, train_loader, valid_loader, criterion, epochs, print_every=40):
    model.to(device)
    model.train()
    steps = 0

    for e in range (epochs): 
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad() 

            outputs = model.forward(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimiser.step() 

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval() 

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, device)

                logging.info(f"Epoch: {e+1}/{epochs}.. ")
                logging.info(f"Training Loss: {(running_loss/print_every):.3f}.. ")
                logging.info(f"Valid Loss: {(valid_loss/len(valid_loader)):.3f}.. ")
                logging.info(f"Valid Accuracy: {(accuracy/len(valid_loader)*100):.3f}%")

                running_loss = 0

                model.train()

def test(model, device, test_loader):
    test_correct = 0
    test_total = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted ==labels).sum().item()

    logging.info(f'Accuracy on test set: {100 * test_correct / test_total}%')


def save(model, path, mapping):
    model.to ('cpu') 

    payload = {
            'classifier': model.classifier,
            'state': model.state_dict(),
            'mapping': mapping
    }        

    torch.save (payload, path)


def prepare_loaders(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    mean_ = [0.485, 0.456, 0.406]
    std_dev_ = [0.229, 0.224, 0.225]

    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean_, std_dev_)
                                            ])

    non_train_transforms = [
                            transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean_, std_dev_)
                            ]

    valid_data_transforms = transforms.Compose(non_train_transforms)
    test_data_transforms = transforms.Compose(non_train_transforms)

    train_image_datasets = datasets.ImageFolder(train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)

    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)

    return train_loader, valid_loader, test_loader, train_image_datasets.class_to_idx

def get_model(arch, hidden_units):
    model = None

    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "vgg13":
        model =  models.vgg13(pretrained=True)
    else:
        raise Exception("Only alexnet and vgg13 are supported.")

    for param in model.parameters(): 
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(9216, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.3)),
                            ('fc2', nn.Linear(hidden_units, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.3)),
                            ('fc3', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim =1))
                            ]))
    model.classifier = classifier

    return model

def get_device(gpu):
    if gpu and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main(args):
    train_loader, valid_loader, test_loader, class_to_idx = prepare_loaders(args.data_directory)
    model = get_model(args.arch, args.hidden_units)
    logging.info(model)

    device = get_device(args.gpu)
    logging.info(f"running on {device}")

    criterion = nn.NLLLoss()
    optimiser = optim.Adam(model.classifier.parameters(), lr =args.learning_rate)
    logging.info(optimiser)

    train(device, model, optimiser, train_loader, valid_loader, criterion, args.epochs)

    test(model, device, test_loader)

    save(model, os.path.join(args.save_dir, "model"), class_to_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='training app')
    parser.add_argument('data_directory', action='store', help='Path to image data')
    parser.add_argument('--save_dir', action='store', dest='save_dir', required=False,
                        default="checkpoints/", help='path to store checkpoints')
    parser.add_argument('--arch', action='store', dest='arch', required=False, 
                        default="alexnet", choices=["alexnet", "vgg13"], help='architecture to use')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', required=False,
                        default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', required=False,
                        default=4096, type=int, help='Number of hidden units')
    parser.add_argument('--epochs', action='store', dest='epochs', required=False,
                        default=7, type=int, help='number of epochs to train for')
    parser.add_argument('--gpu', action='store_true', dest='gpu', required=False,
                        default=False, help='specify for gpu acceleration')

    main(parser.parse_args())