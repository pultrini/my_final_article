# %%
import os
from medmnist import OrganAMNIST, PathMNIST
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader


def data_transform(dataset: str) -> DataLoader:
    '''
    A função faz o download das imagens e retorna os respectivos Dowloads

    args: 
        dataset: "OrganAMNIST" or "PathMNIST",
    return:
        Dataloder do train, val, test do dataset selecionado.
    '''

    transform =  v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.Grayscale(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    DATA_ROOT = 'data'
    os.makedirs(DATA_ROOT, exist_ok=True)
    if dataset == "OrganAMNIST":
        dataset_train = OrganAMNIST(split='train', download=True,
                                root=DATA_ROOT, transform=transform)
        dataset_val = OrganAMNIST(split='val', download=True,
                            root=DATA_ROOT, transform=transform)
        dataset_test = OrganAMNIST(split='test', download=True,
                            root=DATA_ROOT, transform=transform)
    elif dataset == "PathMNIST":
        dataset_train = PathMNIST(split='train', download=True,
                                root=DATA_ROOT, transform=transform)
        dataset_val = PathMNIST(split='val', download=True,
                            root=DATA_ROOT, transform=transform)
        dataset_test = PathMNIST(split='test', download=True,
                            root=DATA_ROOT, transform=transform)
    else:
        raise ValueError("ERROR: Coloque algum dos dois datasets escolhidos 'OrganAMNIST' ou 'PathMNIST'")
    
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=128,
        shuffle=True,
        drop_last=True
    )   

    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )
    return train_loader, val_loader, test_loader