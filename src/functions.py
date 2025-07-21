import torch
from complexity import lmc_complexity

def train(model, dataloader, criterion, optmizer, device,dataset_name, lmc_complexity = lmc_complexity ):
    model = model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for image, label in dataloader:
        if dataset_name == 'PathMNIST':
            image, label = image.to(device), label.squeeze().long().to(device)
        else:
            image, label = image.to(device), label.squeeze().long().to(device)
        optmizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optmizer.step()

        running_loss += loss.item() * image.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted==label).sum().item()
    
    myp = torch.nn.utils.parameters_to_vector(model.parameters())
    myp = myp.cpu().detach().numpy()
    entropy, disequilibrium, complexity = lmc_complexity(myp)
    epoch_loss = running_loss/total
    epoch_acc = correct/total

    return epoch_loss, epoch_acc, entropy, disequilibrium, complexity

def validate(model, dataloader, criterion, device):
    model = model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for image, label in dataloader:
        image, label = image.to(device), label.squeeze().long().to(device)

        outputs = model(image)
        loss = criterion(outputs, label)


        running_loss += loss.item() * image.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted==label).sum().item()
    
    epoch_loss = running_loss/total
    epoch_acc = 100* correct/total

    return epoch_loss, epoch_acc