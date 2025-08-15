# %%
import torch
from torch import nn
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
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                               'nvidia_efficientnet_b0', pretrained=False)


    original_conv_stem = efficientnet.stem.conv
    efficientnet.stem.conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv_stem.out_channels,
        kernel_size=original_conv_stem.kernel_size,
        stride=original_conv_stem.stride,
        padding=original_conv_stem.padding,
        bias = False
    )
    if dataset == 'OrganAMNIST':
        info = INFO['organamnist']
    elif dataset == 'PathMNIST':
        info = INFO['pathmnist']

    n_classes = len(info['label'])
    original_num_features = efficientnet.classifier.fc.in_features

    efficientnet.classifier.fc = nn.Linear(
        in_features=original_num_features,
        out_features=n_classes
    )
    if type == 'loss':
        pre_train_loss = torch.load('model/min_loss.pth')
    if type == 'complexity':
        pre_train_loss = torch.load('model/max_complexity.pth')
    return efficientnet
