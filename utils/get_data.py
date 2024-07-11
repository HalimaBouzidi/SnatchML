import os
import torch
import pickle
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import PIL.Image
from zipfile import ZipFile
import io

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

        self.identity_to_number = {identity: i for i, identity in enumerate(set([item[1] for item in data]))}
        self.class_to_number = {classe: i for i, classe in enumerate(set([item[2] for item in data]))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, identity, classe = self.data[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        identity_number = self.identity_to_number[identity]
        class_number = self.class_to_number[classe]

        identity_tensor = torch.tensor(identity_number)
        class_tensor = torch.tensor(class_number)

        return image, identity_tensor, class_tensor


class OlivettiFacesDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target, target


class MultiLabelTestDataset(Dataset):
    def __init__(self, root, transform=None, skip_class='NORMAL'):
        self.root = root
        self.transform = transform
        self.classes = sorted([cls for cls in os.listdir(root) if cls != skip_class and not cls.startswith('.')])  # Exclude directories starting with '.'
        self.images = []

        for class_label in self.classes:
            class_path = os.path.join(root, class_label)
            class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if not img.startswith('.')]  # Exclude files starting with '.'
            self.images.extend(class_images)

        self.identity_mapping = {f'person{i}': i for i in range(2000)}  # Adjust the range as needed

        self.pathology_mapping = {'bacteria': 0, 'virus': 1}  # Adjust as needed


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        identity, pathology, class_label = os.path.basename(image_path).split('_')[:3]

        class_label = 0 if class_label.lower() == 'n.jpeg' else 1

        pathology_onehot = [self.pathology_mapping.get(pathology.lower(), 0)]
        identity = self.identity_mapping.get(identity, 0)  # Default to 0 if not found

        image = PIL.Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        pathology_onehot = torch.tensor(pathology_onehot, dtype=torch.float32)

        return image, class_label, pathology_onehot

def get_transform():
    transform = transforms.Compose([
    transforms.Resize((48, 48)), 
    transforms.ToTensor(), 
    ])

    return transform

def get_dataset(mini_ck_dir, seed=1):
    image_files = [filename for filename in os.listdir(mini_ck_dir) if filename.endswith(".png")]

    data_list = []
    for image_filename in image_files:
        identity, classe, _ = image_filename[:-4].split('_') 
        image_path = os.path.join(mini_ck_dir, image_filename)
        data_list.append((image_path, identity, classe))

    identity_to_number = {identity: i for i, identity in enumerate(set([item[1] for item in data_list]))}

    updated_data_list = []
    for image_path, identity, classe in data_list:
        new_ID = identity_to_number[identity]
        updated_data_list.append((image_path, new_ID, classe))

    transform = get_transform()
    dataset = CustomDataset(updated_data_list, transform=transform)

    return dataset


def get_dataloader_oliv(dataset, batch_size=32):
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    test_size = total_samples - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, len(train_dataset), len(test_dataset)

def get_dataloader(dataset, batch_size=32):
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

def get_dataloader_fixed(dataset, batch_size=32):

    with open('./indices/train_indices.pkl', 'rb') as file:  # 'rb' denotes 'read binary'
        train_indices = pickle.load(file)

    with open('./indices/val_indices.pkl', 'rb') as file:  # 'rb' denotes 'read binary'
        val_indices = pickle.load(file)

    with open('./indices/test_indices.pkl', 'rb') as file:  # 'rb' denotes 'read binary'
        test_indices = pickle.load(file)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders for train, val, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(train_indices), len(val_indices), len(test_indices)


class CustomDatasetUTK(torch.utils.data.Dataset):
    def __init__(self, images, ages, genders, races, original, hijack, transform=None):
        self.images = images
        self.ages = ages
        self.genders = genders
        self.races = races
        self.transform = transform
        self.original = original
        self.hijack = hijack

    def categorize_age(self, age):
        if age <= 6:  # Group : Baby
            return 0
        elif age <= 12:  # Groupe : Kids
            return 1
        elif age <= 17:  # Groupe : Teen
            return 2
        elif age <= 30:  # Group : Young adult
            return 3
        elif age <= 60:  # Group : Adult
            return 4
        else:  # Group : Old
            return 5

    def categorize_race(self, race):
        return int(race)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        age = self.ages[idx]
        gender = self.genders[idx]
        race = self.categorize_race(self.races[idx])

        age_group = self.categorize_age(age)

        if self.transform:
            image = self.transform(image)

        tasks = {'age': age_group, 'gender': gender, 'race': race}

        return image, tasks[self.original], tasks[self.hijack]

def get_utk_dataset(dataset_path, original, hijack):

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    images = []
    ages = []
    genders = []
    races = []

    with ZipFile(dataset_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in file_list:
            if file_name.startswith('UTKFace/') and file_name.endswith('.jpg'):
                try:
                    img_data = zip_ref.read(file_name)
                    img = Image.open(io.BytesIO(img_data))
                    img = img.convert('RGB')  # Convert to RGB if necessary

                    split = file_name.split('_')
                    age = int(split[0].split('/')[-1])  # Extract the number from the 'UTKFace/100' string
                    gender = int(split[1])
                    race = int(split[2])

                    images.append(img)
                    ages.append(age)
                    genders.append(gender)
                    races.append(race)
                except ValueError as e:
                    pass

    custom_dataset = CustomDatasetUTK(images, ages, genders, races, original, hijack, transform)

    return custom_dataset