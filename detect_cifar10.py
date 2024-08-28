
import os
import math
import random
import argparse
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from utils.resnet_factory import SimSiamWithCls
from resnet18_32x32 import ResNet18_32x32


def confidences_auc(confidences, datasets):

    confidences = np.array(confidences)
    id_confi = confidences[0]

    for (ood_confi, dataset) in zip(confidences[1:], datasets[1:]):

        auroc, aupr_in = auc(id_confi, ood_confi)
        print(f"For {dataset}, AUC: {auroc}")

def search_k(similarities, methods, K=20):

    clean_sim = similarities[0]
    print(f"-------------- K is {K} ----------------")
    for (ae_sim, method) in zip(similarities[1:], methods):

        auroc, aupr_in = auc(clean_sim[:, K-1], ae_sim[:, K-1])
        print(f"For {method}, AUC: {auroc}")

def auc(ind_conf, ood_conf):

    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in



def multi_transform(img, transforms, times=50):

    return torch.stack([transforms(img) for t in range(times)], dim=1)

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

    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        normalization
    ])

    
    # load model and images
    target_model = ResNet18_32x32()
    target_model.load_state_dict(torch.load(f"{args.weights_dir}/{args.model_name}.pth", map_location='cpu'))
    target_model.to(device)
    target_model.eval()

    model = SimSiamWithCls()
    model.load_state_dict(torch.load(f"{args.weights_dir}/{args.ssl_model}.pth"))
    model.to(device)

    backbone = model.backbone
    classifier = model.classifier
    projector = model.projector



    aug_num = args.aug_num
    batch_size = args.bs
    alpha = args.alpha
    auc_metrics = []
    attacks = args.attacks.split(",")
    attacks.insert(0, 'clean')
    K = args.K

    with torch.no_grad():

        target_model.eval()
        backbone.eval()
        classifier.eval()
        projector.eval()

        natural_labels = torch.max(torch.from_numpy(np.load('./AEs/clean_labels.npy')), -1)[1]
        
        
        for attack in attacks:

            attack = attack.strip().upper()

            if attack == "CLEAN":
                samples = torch.from_numpy(np.load('./AEs/clean_inputs.npy'))
                labels = torch.max(torch.from_numpy(np.load('./AEs/clean_labels.npy')), -1)[1]
            elif attack == "FGSM":
                samples = torch.from_numpy(np.load('./AEs/raw/FGSM_AdvSamples.npy'))
                labels = torch.from_numpy(np.load('./AEs/raw/FGSM_AdvLabels.npy'))
            elif attack == 'PGD':
                samples = torch.from_numpy(np.load(f'./AEs/raw/PGD_AdvSamples.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/raw/PGD_AdvLabels.npy'))
            elif attack == "CW":
                samples = torch.from_numpy(np.load('./AEs/raw/CW_AdvSamples.npy'))
                labels = torch.from_numpy(np.load('./AEs/raw/CW_AdvLabels.npy'))
            elif attack in ['APGD-CE', 'APGD-T', 'FAB-T', 'SQUARE']:
                samples = torch.from_numpy(np.load(f'./AEs/raw/{attack}_AdvSamples.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/raw/{attack}_AdvLabels.npy'))
            elif attack == "ADAPTIVE":
                samples = torch.from_numpy(np.load(f'./AEs/adaptive/Ada_AdvSamples.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/adaptive/Ada_AdvLabels.npy'))
                
            else:
                print("Unknown Attacks or Ungenerated Attacks")
                break

            
            print('----------------', attack)
            samples = samples.to(device)

            preds = target_model(samples)
           
            mask = (labels == natural_labels) if attack == 'CLEAN' else (labels != natural_labels)

            samples = samples[mask, :, :, :]
            preds = preds[mask, :]
            labels = labels.masked_select(mask)

            print("Success AEs Num:", len(labels))

            number_batch = int(math.ceil(len(samples) / batch_size))
            
            ssl_repres = []
            aug_repres = []
            ssl_labels = []
            aug_labels = []
            aug_preds = []

            for index in range(number_batch):
                start = index * batch_size
                end = min((index + 1) * batch_size, len(samples))

                trans_images = multi_transform(samples[start:end], img_transforms, times=aug_num).to(device)
                ssl_backbone_out = backbone(normalization(samples[start:end]).to(device))

                ssl_repre = projector(ssl_backbone_out)
                ssl_preds = classifier(ssl_backbone_out)
                ssl_label = torch.max(ssl_preds, -1)[1]

                aug_backbone_out = backbone(trans_images.reshape(-1, 3, 32, 32))
                aug_repre = projector(aug_backbone_out)
                aug_pred = classifier(aug_backbone_out)
                aug_label = torch.max(aug_pred, -1)[1]
                aug_label = aug_label.reshape(end-start, aug_num)
                aug_pred = aug_pred.reshape(end-start, aug_num, -1)

                ssl_labels.append(ssl_label)
                aug_labels.append(aug_label)
                ssl_repres.append(ssl_repre)
                aug_repres.append(aug_repre)
                aug_preds.append(aug_pred)


            ssl_labels = torch.cat(ssl_labels, dim=0)
            aug_labels = torch.cat(aug_labels, dim=0)
            ssl_repres = torch.cat(ssl_repres, dim=0)
            aug_repres = torch.cat(aug_repres, dim=0)
            aug_preds = torch.cat(aug_preds, dim=0)

            sim_repre = F.cosine_similarity(ssl_repres.unsqueeze(dim=1), aug_repres.reshape(len(samples), aug_num, -1), dim=2)
            sim_preds = F.cosine_similarity(F.one_hot(labels, num_classes=10).unsqueeze(dim=1).cuda(), aug_preds.reshape(len(samples), aug_num, -1), dim=2)
            labels = labels.to(device)

            print(f'{attack}, Similarity with Ori Mean: {sim_repre.mean(dim=-1).mean()}, Variance: {sim_repre.var(-1).mean()}')
            print(f'{attack}, Prediction Similarity: {sim_preds.mean(dim=-1).mean()}, Variance: {sim_preds.var(-1).mean()}')
            print(f'{attack}, Target Model Label equals Aug Labels Count: {(labels.unsqueeze(dim=1) == aug_labels).sum(-1).float().mean()}')
            print(f'{attack}, SSL Model Label equals Aug Labels Count: {(ssl_labels.unsqueeze(dim=1) == aug_labels).sum(-1).float().mean()}')


            auc_metrics.append((alpha * sim_preds + (1-alpha)*sim_repre).sort(descending=True)[0].cpu().numpy())

        search_k(auc_metrics, attacks[1:], K)





if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Detection process of BEYOND')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--weights_dir', type=str, default='./weights', help='the directory to store model weights')
    parser.add_argument('--model_name', type=str, default='resnet_c10')
    parser.add_argument('--ssl_model', type=str, default='simsiam_c10')


    parser.add_argument('--attacks', type=str, default='fgsm', help='attack methods: fgsm, pgd, cw, apgd-ce, apgd-t, fab-t, square, adaptive')

    parser.add_argument('--aug_num', type=int, default=50, help='number of augmentation')
    parser.add_argument('--bs', type=int, default=100, help='batch size')
    parser.add_argument('--alpha', type=float, default=0.8, help='1: Label, 0: Representation')
    parser.add_argument('--K', type=int, default=20)

    arguments = parser.parse_args()
    main(arguments)

    # python detect_cifar10.py --attacks=fgsm,pgd,cw,apgd-ce,apgd-t,fab-t,square

