import torch.optim as optim

def train_model(model, train_loader, criterion, T):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    for epoch in range(T):
        correct, total, running_loss = 0, 0, 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
            running_loss += criterion(model(images), labels).item()
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          
        print("Train Epoch: {} Accuracy: {}/{} ({:.2f}%) Loss: {:.3f}".format(
            epoch, correct, total, 100.0 * correct / total, running_loss / len(train_loader)))
