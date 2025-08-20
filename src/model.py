# %%
import torch
from torch import nn
import torchvision
from medmnist import INFO
# %%
def create_model(dataset:str, type:str = None):
    '''
    Retorna o modelo adaptado aos devidos datasets que setão utilizados.
    args:
        dataset: Banco de dados escolhido
            "organamnist" ou "pathmnist"
    return:
        modelo.
    '''
    #trocar o modelo para uma com menos convolucionais 
    model = torchvision.models.resnet50(weights=None)


    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,  # agora aceita imagens com 1 canal (ex: grayscale)
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    if dataset == 'BloodMNIST':
        info = INFO['bloodmnist']
    elif dataset == 'TissueMNIST':
        info = INFO['tissuemnist']

    n_classes = len(info['label'])
    original_num_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features=original_num_features,
        out_features=n_classes
    )
    if type == 'loss':
        pre_train_loss = torch.load('/home/users/u12559743/Documentos/my_final_article/models/min_loss.pth')
        model.load_state_dict(pre_train_loss)
    if type == 'complexity':
        pre_train_complexity = torch.load('/home/users/u12559743/Documentos/my_final_article/models/max_complexity.pth')
        model.load_state_dict(pre_train_complexity)
    return model

