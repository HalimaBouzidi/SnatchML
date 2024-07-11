import os, random
import csv, torch
import torch.nn as nn
import torch.optim as optim
from utils.get_data import MultiLabelTestDataset
from utils.get_model_pneu import *
from sklearn.metrics.pairwise import cosine_similarity

import torch
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def find_min_indices(arr, k):
    if k > len(arr):
        raise ValueError("k cannot be larger than the array size")
    idx = np.argpartition(arr, k)
    sorted_idx = idx[:k].argsort()

    return idx[sorted_idx]

def find_max_indices(arr, k):
    if k > len(arr):
        raise ValueError("k cannot be larger than the array size")

    idx = np.argpartition(arr, -k)[-k:]
    sorted_idx = np.argsort(arr[idx])[::-1]

    return idx[sorted_idx]

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

import argparse

parser = argparse.ArgumentParser(description='Test SnatchML hijacking attack in Pneumonia scenario')
parser.add_argument('--seed', default=1, type=int, help='Value of the random seed.')
parser.add_argument('--model', default='simple', type=str, choices=['simple', 'resnet', 'mobilenet', 'transformer'])
parser.add_argument('--setting', default='black', type=str, help='Specify the attack setting', choices=['black', 'white'])
parser.add_argument('--expand', default=1.0, type=float, help='Width expand ratio')
parser.add_argument('--idx', default=0, type=int, help='idx')

run_args = parser.parse_args()

if __name__ == '__main__':

    set_random_seeds(random_seed=int(run_args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = run_args.model # 'resnet' 'transformer' 'mobilenet' 'simple'
    
    path = "./datasets/chest_xray/"
    transformers = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    categories = ['train','val','test']
    dset = {x : torchvision.datasets.ImageFolder(path+x, transform=transformers) for x in categories}

    dataset_sizes = {x : len(dset[x]) for x in categories}
    dataloaders =  {x : torch.utils.data.DataLoader(dset[x], batch_size=16, shuffle=True, num_workers=0) for x in categories}

    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    len_train, len_val, len_test = dataset_sizes['train'], dataset_sizes['val'], dataset_sizes['test']

    if model_name == 'simple':
        model = SimpleModel(in_channels=3, num_classes=2, expand=float(run_args.expand))
    elif model_name == 'mobilenet':
        model = MobileNetV2(in_channels=3, num_classes=2, expand=float(run_args.expand))
    elif model_name == 'resnet':
        model = ResNet(in_channels=3, num_classes=2, expand=float(run_args.expand))
    elif model_name == 'transformer':
        model = TransformerModel(in_channels=3, num_classes=2, expand=float(run_args.expand))
    else:
        raise NotImplementedError
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    model, base_acc, base_loss = train_model(model, num_epochs, optimizer, criterion, \
                        train_loader, val_loader, test_loader, len_train, len_val, len_test, device)
    
    print(f"Top-1 Original Task Accuracy: {base_acc:.4f}")
    
    if run_args.setting == 'black':

        if model_name == 'resnet':
            desired_submodel = SubModelN(model, n_layers=10, model_name=model_name, setting=run_args.setting)
        elif model_name == 'mobilenet':
            desired_submodel = SubModelN(model, n_layers=2, model_name=model_name, setting=run_args.setting)
        elif model_name == 'transformer':
            desired_submodel = SubModelN(model, n_layers=2, model_name=model_name, setting=run_args.setting)
        elif model_name == 'simple':
            desired_submodel = copy.deepcopy(model)

    elif run_args.setting == 'white':

        if model_name == 'resnet':
            desired_submodel = SubModelN(model, n_layers=9, model_name=model_name, setting=run_args.setting)
        elif model_name == 'mobilenet':
            desired_submodel = SubModelN(model, n_layers=1, model_name=model_name, setting=run_args.setting)
        elif model_name == 'transformer':
            desired_submodel = SubModelN(model, n_layers=2, model_name=model_name, setting=run_args.setting)
            desired_submodel.layer0.heads = nn.Identity()
        elif model_name == 'simple':
            desired_submodel = copy.deepcopy(model)
            desired_submodel.fc_2 = nn.Identity()
            desired_submodel.relu = nn.Identity()
    else:
        raise NotImplementedError

    desired_submodel.to(device)
    desired_submodel.eval()


    test_path = os.path.join(path, 'test')
    skip_class = 'NORMAL'
    multi_label_test_dataset = MultiLabelTestDataset(test_path, transform=transformers, skip_class=skip_class)

    multi_label_test_loader = DataLoader(multi_label_test_dataset, batch_size=16, shuffle=False, num_workers=0)
    unique_identities = set()
    
    real_image_list = []
    real_output_list = []
    real_pathology_list = []


    for images, classes, pathos in multi_label_test_loader:
        images = images.to(device)
        classes = classes.to(device).long()
        pathos = pathos.to(device).long()

        with torch.no_grad():
            output = desired_submodel(images)

        real_image_list.append(images)
        real_output_list.append(output)
        real_pathology_list.append(pathos)

    real_all_images = torch.cat(real_image_list, dim=0)
    real_all_outputs = torch.cat(real_output_list, dim=0)
    real_all_pathology = torch.cat(real_pathology_list, dim=0)

    distances = cosine_similarity(real_all_outputs.detach().cpu(), real_all_outputs.detach().cpu())
    np.fill_diagonal(distances, float('-inf'))
    selector = find_max_indices
    virtual_top_accuracies = []  # Define an empty list to store accuracies

    correct = 0
    correct_indices = []
    correct_images = []

    for id_, elem in enumerate(distances, start=0):
        indices = selector(elem, 1)
        candidates = [real_all_pathology[indices[i]].item() for i in range(len(indices))] # to check if it's in top-k
        if real_all_pathology[id_].item() in candidates:
            correct_indices.append(indices)
            correct_images.append(id_)
            correct += 1

    accuracy = correct / len(real_all_outputs)
    virtual_top_accuracies.append(accuracy)
    print(f"Top-1 Hijacking Task Accuracy: {accuracy:.4f}")

    file_name = 'pneumonia_type_detection.csv'
    with open('./results/'+file_name, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([run_args.model, run_args.setting, run_args.expand, run_args.seed, base_acc, base_loss]+virtual_top_accuracies)
    