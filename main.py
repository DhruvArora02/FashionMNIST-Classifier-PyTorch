import torch
import torch.nn as nn
from data_loader import get_data_loader
from model import build_model
from trainer import train_model
from evaluator import evaluate_model
from predictor import predict_label

def main():
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)
    model = build_model()
    
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    
    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 0)

if __name__ == '__main__':
    main()
