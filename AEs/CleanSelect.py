import numpy as np
import argparse
import random

import torch
from torchvision import transforms, datasets

import os
import sys 
sys.path.append("..") 
from resnet18_32x32 import ResNet18_32x32


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num = args.number

    
    model = ResNet18_32x32().to(device)
    model.load_state_dict(torch.load(args.weights_dir, map_location='cuda'))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.CIFAR10(root=args.data_dir, 
        train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_size=1, shuffle=False, num_workers=2)

    successful = []
    model.eval()

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            if predicted == label:
                successful.append([image, label])
    
    print(len(successful))
    candidates = random.sample(successful, num)

    candidate_images = []
    candidate_labels = []

    for index in range(len(candidates)):
        image = candidates[index][0].cpu().numpy()
        image = np.squeeze(image, axis=0)
        candidate_images.append(image)

        label = candidates[index][1].cpu().numpy()[0]


        one_hot_label = [0 for i in range(10)]
        one_hot_label[label] = 1

        candidate_labels.append(one_hot_label)

    candidate_images = np.array(candidate_images)
    candidate_labels = np.array(candidate_labels)
    
    np.save('./clean_inputs.npy', candidate_images)
    np.save('./clean_labels.npy', candidate_labels)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Candidate Selection for Clean Data set')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--number', type=int, default=1000, help='the total number of candidate samples that will be randomly selected')
    parser.add_argument('--weights_dir', type=str, default='../weights/resnet_c10.pth')
    parser.add_argument('--data_dir', type=str, help='the path of dataset')

    arguments = parser.parse_args()
    main(arguments)

    # python CleanSelect.py --data_dir=../../../datasets/