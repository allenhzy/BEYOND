import os
import copy
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from resnet18_32x32 import ResNet18_32x32



def train_step(model,features,labels, criterion, optimizer):
    
    model.train()
    
    optimizer.zero_grad()
    
    predictions = model(features)
    loss = criterion(predictions,labels)
    _, pred_labels = torch.max(predictions, 1)
    acc = (pred_labels == labels).float().mean()

    loss.backward()
    optimizer.step()

    return loss.item(),acc.item()

def valid_step(model,features,labels, criterion):
    
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = criterion(predictions,labels)
        _, pred_labels = torch.max(predictions, 1)
        acc = (pred_labels == labels).float().mean()
    
    return loss.item(), acc.item()



def train_model(model, criterion, dl_train, dl_valid,optimizer, device, num_epochs=200, log_step_freq=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Start Training.............")
    model.to(device)
    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        loss_sum = 0.
        acc_sum = 0.

        for step, (features,labels) in enumerate(dl_train, 1):
            
            features, labels = features.to(device), labels.to(device)
            loss,acc = train_step(model,features,labels, criterion, optimizer)

            loss_sum += loss
            acc_sum += acc
            if step%log_step_freq == 0:   
                print(f"[step = {step}] loss: {loss_sum/step:.6f}, acc: {acc_sum/step:.4f}")

        # scheduler.step()
        
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            features, labels = features.to(device), labels.to(device)
            val_loss,val_acc = valid_step(model,features,labels,criterion)

            val_loss_sum += val_loss
            val_acc_sum += val_acc
        
        print(f"\nEPOCH = {epoch}, loss = {loss_sum/step:.6f}, acc = {acc_sum/step:.4f}, \
            val_loss = {val_loss_sum/val_step:.6f}, val_acc = {val_acc_sum/val_step:.4f}") 

        if val_acc_sum/val_step > best_acc:
                best_acc = val_acc_sum/val_step
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return best_model_wts

def main(args):

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = args.epoch
    batch_size = args.bs
    learning_rate = args.lr


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                train=True, 
                                                transform=transform_train,
                                                download=True)

    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                train=False, 
                                                transform=transform_test,
                                                download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=args.nw)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=args.nw)

    model = ResNet18_32x32()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_wts = train_model(model, criterion, train_loader, test_loader, optimizer, device, num_epochs)
    torch.save(best_model_wts, f"{args.weights_dir}/{args.model_name}.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train ResNet18 on Cifar10')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--data_dir', type=str, help='the path of dataset')
    parser.add_argument('--weights_dir', type=str, default='./weights/', help='the directory to store model weights')
    parser.add_argument('--model_name', type=str, default='resnet_c10')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--nw', type=int, default=2, help='number of worker')



    arguments = parser.parse_args()
    main(arguments)

    # python train_resnet18.py --data_dir=../../datasets/



