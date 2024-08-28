import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from resnet18_32x32 import ResNet18_32x32

from autoattack import AutoAttack




def main(args):

    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load model and images
    model = ResNet18_32x32().to(device)
    model.load_state_dict(torch.load(f"{args.weights_dir}/{args.model_name}.pth", map_location='cpu'))
    model.to(device)
    model.eval()

    nature_samples = Variable(torch.from_numpy(np.load('./AEs/clean_inputs.npy')).to(device), requires_grad=True)
    labels_samples = Variable(torch.LongTensor(np.load('./AEs/clean_labels.npy')).to(device), requires_grad=False)
    labels_samples = labels_samples.max(1)[1]


    pred_labels = model(nature_samples)
    pred_labels = torch.max(pred_labels, 1)[1]

    acc = (pred_labels == labels_samples).sum() / len(labels_samples)
    print(f"Model Accuracy is {acc.item()*100:.2f}")

    epsilon = args.e
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)

    # adv_samples, adv_labels = adversary.run_standard_evaluation(nature_samples, labels_samples, bs=64, return_labels=True)
    attack = args.attack
    print(f"Attack: {attack.upper()}")
    if attack in ['apgd-ce', 'apgd-t', 'fab-t', 'square']:

        adversary.attacks_to_run = [attack] # apgd-ce, apgd-t, fab-t, square
        adv_dict = adversary.run_standard_evaluation_individual(nature_samples, labels_samples, bs=args.bs, return_labels=True)
        for k,v in adv_dict.items():
            # print(k)
            adv_samples, adv_labels = v

            np.save(f'./AEs/raw/{k.upper()}_AdvSamples.npy', adv_samples.detach().cpu().numpy())
            np.save(f'./AEs/raw/{k.upper()}_AdvLabels.npy', adv_labels.detach().cpu().numpy())

            print(f"For {k.upper()}, Attack success rate is {(labels_samples != adv_labels).sum() / len(adv_labels) * 100:.2f}")
    else:
        print("Unknown Attack!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Auto Attacks')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--weights_dir', type=str, default='./weights', help='the directory to store model weights')
    parser.add_argument('--model_name', type=str, default='resnet_c10')

    parser.add_argument('--attack', type=str, default='apgd-ce', help='attack methods: apgd-ce, apgd-t, fab-t, square')

    parser.add_argument('--e', type=float, default=8/255., help='perturbation budget')
    parser.add_argument('--bs', type=int, default=256, help='batch size')

    arguments = parser.parse_args()
    main(arguments)

    # python auto_attack.py --attack=apgd-ce 