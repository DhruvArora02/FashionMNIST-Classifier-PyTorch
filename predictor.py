import torch
import torch.nn.functional as F

def predict_label(model, test_images, index):
    model.eval()
    
    with torch.no_grad():
        logits = model(test_images)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs[index], 3)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    for i in range(3):
        print("{}: {:.2f}%".format(class_names[top_indices[i]], top_probs[i].item() * 100))
