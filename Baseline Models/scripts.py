import torch
import torchvision
from utils import printf

def train_model(model, loader, optimizer, n_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, train_acc, test_acc = [], [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_correct, n_total = 0, 0
        for i, (features, labels) in enumerate(loader):
            features, labels = features.to(device), labels.to(device)
            
            output = model(features)
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            correct = (output.argmax(dim = -1) == labels).sum().item()
            n_correct += correct
            n_total += len(labels)
            
            if (i + 1) % 20 == 0:
                printf(f'Epoch: [{epoch+1}/{n_epochs}], Batch: [{i+1}/{len(loader)}], Loss: {loss.item():.4f}, Training Accuracy: {correct/len(labels):.4f}')
        
        epoch_loss /= len(loader)    
        
        train_loss.append(epoch_loss)
        train_acc.append(n_correct / n_total)
    
    print() # for prettiness
    
    results = {
        'epochs': torch.arange(1, n_epochs + 1),
        'train losses': train_loss,
        'train accuracies': train_acc
    }
    
    return model, results

def test_model(model, loader, device):
    with torch.no_grad():
        model.eval()
        n_correct, n_total = 0, 0
        
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            output = output.argmax(dim=-1)
            n_correct += (output == labels).sum().item()
            n_total += len(labels)
    
        return n_correct / n_total