import torch
import torch.nn.functional as F

def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    
    with torch.no_grad():
        
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if show_loss:
        print("Average loss: {:.4f}".format(test_loss / len(test_loader)))
    
    print("Accuracy: {:.2f}%".format(100.0 * correct / total))
