import copy, random
import csv, torch
import torch.nn as nn
import torch.optim as optim
from utils.get_data import *
from utils.get_model_unlearn import *
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from torchvision import transforms
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, Dataset

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

parser = argparse.ArgumentParser(description='Test SnatchML hijacking attack in ER scenario')
parser.add_argument('--seed', default=1, type=int, help='Value of the random seed.')
parser.add_argument('--model', default='simple', type=str, choices=['simple', 'resnet', 'mobilenet', 'transformer'])
parser.add_argument('--setting', default='black', type=str, help='Specify the attack setting', choices=['black', 'white'])
parser.add_argument('--hijack-dataset', default='olivetti', type=str, help='Specify the hijacking dataset', choices=['olivetti', 'celebrity', 'synthetic'])
parser.add_argument('--expand', default=1.0, type=float, help='Width expand ratio')
parser.add_argument('--idx', default=0, type=int, help='idx')

run_args = parser.parse_args()

if __name__ == '__main__':

    set_random_seeds(random_seed=int(run_args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = run_args.model # 'resnet' 'transformer' 'mobilenet' 'simple'
    mini_ck_dir = './datasets/mini_3_ck/'
    dataset = get_dataset(mini_ck_dir)
    train_loader, val_loader, test_loader, len_train, len_val, len_test = get_dataloader_fixed(dataset, batch_size=32)

    if model_name == 'simple':
        model = SimpleModel(in_channels=1, num_classes=[6, 98], expand=float(run_args.expand))
    elif model_name == 'mobilenet':
        model = MobileNetV2(in_channels=1, num_classes=[6, 98], expand=float(run_args.expand))
    elif model_name == 'resnet':
        model = ResNet(in_channels=1, num_classes=[6, 98], expand=float(run_args.expand))
    elif model_name == 'transformer':
        model = TransformerModel(in_channels=1, num_classes=[6, 98], expand=float(run_args.expand))
    else:
        raise NotImplementedError
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model, base_acc, base_loss = train_model_er(model, num_epochs, optimizer, criterion, \
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

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((48, 48)), transforms.ToTensor(),])

    if run_args.hijack_dataset == 'olivetti':

        faces = fetch_olivetti_faces(download_if_missing=True)
        data = torch.tensor(faces.images)
        targets = torch.tensor(faces.target)
        olivetti_dataset = OlivettiFacesDataset(data, targets, transform=transform)
        test_loader = DataLoader(olivetti_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        file_name = 'unlearn_identification_oliv.csv'

    elif run_args.hijack_dataset == 'celebrity':
        real_dir = './datasets/real_face_exp_gen_gray'
        real_dataset = get_dataset(real_dir)
        test_loader = DataLoader(real_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        file_name = 'unlearn_identification_celebrity.csv'

    elif run_args.hijack_dataset == 'synthetic':
        real_dir = './datasets/gen_grayscale'
        real_dataset = get_dataset(real_dir)
        test_loader = DataLoader(real_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        file_name = 'unlearn_identification_synthetic.csv'

    else:
        file_name = 'unlearn_reidentification_ck.csv'
    
    real_image_list = []
    real_output_list  = []
    real_ids_list  = []

    for image, ids, _ in test_loader:
        image = image.to(device)
        ids = ids.to(device)
        output = desired_submodel(image)

        real_image_list.append(image)
        real_output_list.append(output)
        real_ids_list.append(ids)

    real_all_images = torch.cat(real_image_list, dim=0)
    real_all_outputs = torch.cat(real_output_list, dim=0)
    real_all_ids = torch.cat(real_ids_list, dim=0)

    mesure = 'cosine'  #'cosine' #'euclidean', 'kl-diver'
    distances = cosine_similarity(real_all_outputs.detach().cpu(), real_all_outputs.detach().cpu())
    np.fill_diagonal(distances, float('-inf'))
    selector = find_max_indices

    real_top_accuracies = [] 
    for k in range(1, 6):
        correct = 0
        correct_indices = []
        correct_images = []

        for id_, elem in enumerate(distances, start=0):
            indices = selector(elem, k)
            candidates = [real_all_ids[indices[i]].item() for i in range(len(indices))] 
            if real_all_ids[id_].item() in candidates:
                correct_indices.append(indices)
                correct_images.append(id_)
                correct += 1

        accuracy = correct / len(real_all_outputs)
        real_top_accuracies.append(accuracy)
        print(f"Top-{k} Hijacking Task Accuracy: {accuracy:.4f}")
   
    with open('./results/'+file_name, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([run_args.model, run_args.setting, run_args.expand, run_args.seed, base_acc, base_loss]+real_top_accuracies)