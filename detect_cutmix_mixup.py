# -*- encoding: utf-8 -*-
'''
@File: 3090_detect_adv_independent.py
@Description: Detection AEs by SSL representation
@Time: 2022/07/03 16:12:13
@Author: Zhiyuan He
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import v2

from utils.resnet_factory import SimSiamWithCls
from resnet18_32x32 import ResNet18_32x32


import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import random
import os
import math
import argparse

# CutMix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
 
    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    #限制坐标区域不超过样本大小
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2

def mask_image(image, mask_size=8, stride=8):
    
    images = []
    b, c, h, w = image.shape
    cur_h = 0
    while cur_h <= h-mask_size:
        cur_w = 0
        while cur_w <= w-mask_size:
            
            mask = torch.zeros_like(image)
            mask[:,:,cur_h:cur_h+mask_size,cur_w:cur_w+mask_size] = 1
            mask_img = image.masked_fill(mask==1, 0.)
            images.append(mask_img)
            cur_w += stride

        cur_h += stride
    return torch.stack(images, dim=1)

def confidences_auc(confidences, datasets):

    confidences = np.array(confidences)
    id_confi = confidences[0]

    for (ood_confi, dataset) in zip(confidences[1:], datasets[1:]):

        auroc, aupr_in = auc(id_confi, ood_confi)
        print(f"For {dataset}, AUC: {auroc}")

def search_k(confidences, datasets, K=10):

    confidences = np.array(confidences)
    id_confi = confidences[0]

    for i in range(id_confi.shape[1]):
        
        print(f"-------------- K is {i+1} ----------------")
        for (ood_confi, dataset) in zip(confidences[1:], datasets[1:]):

            auroc, aupr_in = auc(id_confi[:, i], ood_confi[:, i])
            print(f"For {dataset}, AUC: {auroc}")

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


def plot_roc(auc_metrics:list, labels:list, path:str):

    results = []
    auc_metrics = np.array(auc_metrics)
    
    plt.figure()
    x = auc_metrics[0]

    x_sort_index = x.argsort()
    x = np.append(np.insert(x[x_sort_index], 0, 0.), 1.0)
    results.append(x)
    ys = auc_metrics[1:]
    for (y, label) in zip(ys, labels):
        y = np.append(np.insert(y[x_sort_index], 0, 0.), 1.0)
        results.append(y)
        plt.plot(x, y, lw=2, label=label)
        print(f"For {label}, AUC: {metrics.auc(x, y)}")
    plt.plot([0, 1],[0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel( 'False Positive Rate')
    plt.ylabel( 'True Positive Rate')
    plt.title( 'Receiver operating characteristic example' )
    plt.legend (loc="lower right")
    # plt.savefig(path)

    return results


def multi_transform(img, labels, transforms, times=50):

    return torch.stack([transforms(img, labels)[0] for t in range(times)], dim=1)

def get_cos_similarity_3d(m1, m2):

    n = torch.matmul(m1, m2.transpose(1,2))
    dnorm = torch.norm(m1,p=2, dim=2).unsqueeze(dim=2) * torch.norm(m2, p=2, dim=2).unsqueeze(dim=1)

    res = n/dnorm
    return res

def detect_adv_by_representation(tar_labels, aug_labels, sim_with_ori):

    # adv_num = 0
    total_num = len(tar_labels)

    thresholds = [(0.15, 25), (0.15, 30.0), (0.2, 35.0),
        (0.25, 20.0), (0.25, 35.0), (0.25, 30.0),
        (0.3, 30), (0.3, 35.0), 
        (0.4, 35.0), (0.4, 30.0), (0.4, 25.0),
        (0.5, 35.0), (0.5, 30.0),
        (0.6, 35.0), (0.6, 25.0),
        (0.7, 35.0), (0.7, 30.0), (0.7, 25.0),
        (0.8, 35.0), (0.8, 30.0), (0.8, 25.0),
        (0.9, 35.0),
        (0.9, 30.0), (0.9, 25.0), (0.9, 15.0), (0.9, 5.0)]

    auc_metrics = []
    for t_sim, t_c in thresholds:
    # for t in thresholds:
        # print("-"*20)
        sim_lt_t = (sim_with_ori < t_sim).sum(-1)

        # for c in counts:
        sim_lt_adv = (sim_lt_t > t_c).sum()
        adv_single_sam = sim_lt_adv.item()
        print(adv_single_sam/total_num)
        auc_metrics.append(adv_single_sam/total_num)

    return auc_metrics


def detect_adv_by_label_sim(tar_labels, aug_labels, aug_time):
    
    adv_num = 0
    total_num = len(tar_labels)
    aug_labels_mode = aug_labels.mode(dim=1)[0]
    aug_ne_tar = (aug_labels_mode != tar_labels)
    # aug_ne_ssl = (aug_labels_mode != ssl_labels)

    # adv_num += aug_ne_tar.sum().item()
    print(f'First Detection: {aug_ne_tar.sum().item()}/{total_num}')

    aug_labels = aug_labels.reshape(total_num, aug_time)

    aug_eq_tar = (aug_labels == tar_labels.unsqueeze(dim=1))
    # print(aug_eq_tar.sum(dim=-1).float().mean())

    auc_metrics = []
    for threshold in range(0, aug_time+1, 1):

        aug_tar_lt_threshold = (aug_eq_tar.sum(dim=-1)<threshold).sum()
        auc_metrics.append((aug_tar_lt_threshold.item())/total_num)

    print("detect between target label and aug labels.")
    print(auc_metrics[11:15])

    return auc_metrics

def detect_adv_by_label_rep(tar_labels, aug_labels, sim_with_ori, aug_time):
    

    total_num = len(tar_labels)
    
    aug_labels = aug_labels.reshape(total_num, aug_time)
    aug_eq_tar = (aug_labels == tar_labels.unsqueeze(dim=1))

    auc_metrics = []

    rep_thresholds = [(0.15, 25), (0.15, 30.0), (0.2, 35.0),
        (0.25, 20.0), (0.25, 35.0), (0.25, 30.0),
        (0.3, 30), (0.3, 35.0), 
        (0.4, 35.0), (0.4, 30.0), (0.4, 25.0),
        (0.5, 35.0), (0.5, 30.0),
        (0.6, 35.0), (0.6, 25.0),
        (0.7, 35.0), (0.7, 30.0), (0.7, 25.0),
        (0.8, 35.0), (0.8, 30.0), (0.8, 25.0),
        (0.9, 35.0),
        (0.9, 30.0), (0.9, 25.0), (0.9, 15.0), (0.9, 5.0)]
    
    label_thresholds = list(range(0, aug_time+1, 2))

    thresholds = zip(label_thresholds, rep_thresholds)
    for t_l, (t_r_sim, t_r_c) in thresholds:
        
        label_detect = aug_eq_tar.sum(dim=-1)<t_l
        sim_lt_t = (sim_with_ori < t_r_sim).sum(-1)
        rep_detect = (sim_lt_t > t_r_c)

        detect_num = torch.logical_or(label_detect, rep_detect).sum().item()
        auc_metrics.append(detect_num / total_num)
    print(auc_metrics)

    return auc_metrics


def detect_adv_by_rep_label(tar_labels, aug_labels, sim_with_ori, aug_time):
    
    adv_num = 0
    total_num = len(tar_labels)
    aug_labels_mode = aug_labels.mode(dim=1)[0]
    aug_ne_tar = (aug_labels_mode != tar_labels)
    # aug_ne_ssl = (aug_labels_mode != ssl_labels)

    adv_num += aug_ne_tar.sum().item()
    print(f'First Detection: {adv_num}/{total_num}')

    tar_labels = tar_labels[~aug_ne_tar]
    # ssl_labels = ssl_labels[~aug_ne_tar]
    aug_labels = aug_labels.reshape(total_num, aug_time)[~aug_ne_tar, :]
    sim_with_ori = sim_with_ori[~aug_ne_tar, :]

    aug_eq_tar = (aug_labels == tar_labels.unsqueeze(dim=1))
    # print(aug_eq_tar.sum(dim=-1).float().mean(), aug_eq_ssl.sum(dim=-1).float().mean())


    auc_metrics = []
    thresholds = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    counts = [10, 15, 20, 25, 30, 35]
    # thresholds = [(t, c) for t in thresholds for c in counts]

    thresholds = [(0.15, 25), (0.15, 30.0), (0.2, 35.0),
        (0.25, 20.0), (0.25, 35.0), (0.25, 30.0),
        (0.3, 30), (0.3, 35.0), 
        (0.4, 35.0), (0.4, 30.0), (0.4, 25.0),
        (0.5, 35.0), (0.5, 30.0),
        (0.6, 35.0), (0.6, 25.0),
        (0.7, 35.0), (0.7, 30.0), (0.7, 25.0),
        (0.8, 35.0), (0.8, 30.0), (0.8, 25.0),
        (0.9, 35.0),
        (0.9, 30.0), (0.9, 25.0), (0.9, 15.0), (0.9, 5.0)]
    auc_metrics = []
    for t_sim, t_c in thresholds:
    # for t in thresholds:
        # print("-"*20)
        sim_lt_t = (sim_with_ori < t_sim).sum(-1)
        # print(t, sim_lt_t.float().mean().item())

        # for c in counts:
        sim_lt_adv = (sim_lt_t > t_c).sum()

        adv_single_sam = sim_lt_adv.item()+adv_num
        print(adv_single_sam/total_num)
        auc_metrics.append(adv_single_sam/total_num)

    return auc_metrics

def main():

    seed = 100
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
    target_model.load_state_dict(torch.load('./weights/resnet18_9554.pth'))
    target_model.to(device)
    target_model.eval()

    model = SimSiamWithCls()
    model.load_state_dict(torch.load('./weights/simsiam-cifar10.pth'))
    model.to(device)

    backbone = model.backbone
    classifier = model.classifier
    projector = model.projector

    cutmix = v2.CutMix(alpha=0.7, num_classes=10)
    mixup = v2.MixUp(alpha=0.7, num_classes=10)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    aug_time = 50
    batch_size = 100
    auc_metrics = []
    # attacks = ['clean', 'APGD-CE']
    # attacks = ['clean', 'ada_a-1_e8', 'ada_a0_e8']

    attacks = ['clean', 'fgsm', 'pgd_8', 'cw', 'APGD-CE']
    # attacks = ['clean', 'fgsm', 'pgd_8', 'cw', 'APGD-CE', 'APGD-T', 'FAB-T', 'SQUARE']
    # attacks = ['clean', 'pgd_2', 'pgd_4', 'pgd_8', 'pgd_16', 'pgd_32', 'pgd_64', 'pgd_128']
    # attacks = ['clean', 'select_opgd_1', 'select_opgd_3', 'orth_opgd_1', 'orth_opgd_3']

    # attacks = ['clean', 'ada_a0_e8', 'ada_a-1_e8', 'ada_a-10_e8', 'ada_a-20_e8', 'ada_a-50_e8', 'ada_a-100_e8']
    # attacks = ['clean', 'ada_a-1_e2', 'ada_a-1_e4', 'ada_a-1_e8', 'ada_a-1_e16', 'ada_a-1_e32', 'ada_a-1_e64', 'ada_a-1_e128']
    with torch.no_grad():

        target_model.eval()
        backbone.eval()
        classifier.eval()
        projector.eval()
        
        for attack in attacks:

            if attack == "clean":
                samples = torch.from_numpy(np.load('./AEs/clean_inputs.npy'))
                labels = torch.max(torch.from_numpy(np.load('./AEs/clean_labels.npy')), -1)[1]
            elif attack == "fgsm":
                samples = torch.from_numpy(np.load('./AEs/raw/FGSM_AdvSamples_8.npy'))
                labels = torch.from_numpy(np.load('./AEs/raw/FGSM_AdvLabels_8.npy'))
                # samples = torch.from_numpy(np.load('./AEs/compare/FGSM_AdvSamples_005.npy'))
                # labels = torch.from_numpy(np.load('./AEs/compare/FGSM_AdvLabels_005.npy'))
            elif attack.startswith('pgd_'):
                epsilon = attack.split('_')[1]
                samples = torch.from_numpy(np.load(f'./AEs/raw/PGD_AdvSamples_{epsilon}.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/raw/PGD_AdvLabels_{epsilon}.npy'))
                # samples = torch.from_numpy(np.load(f'./AEs/compare/PGD_AdvSamples_002.npy'))
                # labels = torch.from_numpy(np.load(f'./AEs/compare/PGD_AdvLabels_002.npy'))
            elif attack == "cw":
                samples = torch.from_numpy(np.load('./AEs/raw/CW_AdvSamples_8.npy'))
                labels = torch.from_numpy(np.load('./AEs/raw/CW_AdvLabels_8.npy'))
            elif attack in ['APGD-CE', 'APGD-T', 'FAB-T', 'SQUARE']:
                samples = torch.from_numpy(np.load(f'./AEs/raw/{attack}_AdvSamples_Linf.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/raw/{attack}_AdvLabels_Linf.npy'))
            elif attack.startswith("ada_"):
                alpha = attack.split('_')[1][1:]
                e = attack.split('_')[2][1:]
                # samples = torch.from_numpy(np.load(f'./AEs/ssl/Ada_rep&label_{alpha}_{e}_at{aug_time}_norm_s0002_AdvSamples.npy'))
                # labels = torch.from_numpy(np.load(f'./AEs/ssl/Ada_rep&label_{alpha}_{e}_at{aug_time}_norm_s0002_tar_AdvLabels.npy'))

                samples = torch.from_numpy(np.load(f'./AEs/ssl/Ada_a{alpha}_e{e}_at50_norm_s0002_AdvSamples.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/ssl/Ada_a{alpha}_e{e}_at50_norm_s0002_AdvLabels.npy'))
            elif 'opgd' in attack:
                mode = attack.split('_')[0]
                e = attack.split('_')[-1]
                samples = torch.from_numpy(np.load(f'./AEs/ssl/Opgd_{mode}_tar_e00{e}_AdvSamples.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/ssl/Opgd_{mode}_e00{e}_AdvLabels.npy'))
            elif attack == 'sgm':
                samples = torch.from_numpy(np.load(f'./AEs/ssl/SGM_e8_s0005_rand_AdvSamples.npy'))
                labels = torch.from_numpy(np.load(f'./AEs/ssl/SGM_e8_s0005_rand_AdvLabels.npy'))



                
            else:
                print("Unknown Attacks")
                break

            
            print('----------------', attack)
            samples = samples.to(device)
            natural_labels = torch.max(torch.from_numpy(np.load('./AEs/clean_labels.npy')), -1)[1]
           
            mask = (labels == natural_labels) if attack == 'clean' else (labels != natural_labels)

            samples = samples[mask, :, :, :]
            labels = labels.masked_select(mask)

            print("Success AEs Num:", len(labels))

            number_batch = int(math.ceil(len(samples) / batch_size))
            sim_with_ori = torch.Tensor().to(device)
            sim_representation = torch.Tensor().to(device)

            ssl_repres = torch.Tensor().to(device)
            aug_repres = torch.Tensor().to(device)
            ssl_labels = torch.Tensor().to(device)
            aug_labels = torch.Tensor().to(device)

            for index in range(number_batch):
                start = index * batch_size
                end = min((index + 1) * batch_size, len(samples))

                # trans_images, _ = cutmix(samples[start:end], labels[start:end])

                trans_images = multi_transform(samples[start:end], labels[start:end], cutmix, times=aug_time).to(device)

                ssl_backbone_out = backbone(normalization(samples[start:end]).to(device))

                ssl_repre = projector(ssl_backbone_out)
                ssl_label = classifier(ssl_backbone_out)
                ssl_label = torch.max(ssl_label, -1)[1]

                aug_backbone_out = backbone(trans_images.reshape(-1, 3, 32, 32))
                aug_repre = projector(aug_backbone_out)
                aug_label = classifier(aug_backbone_out)
                aug_label = torch.max(aug_label, -1)[1]
                aug_label = aug_label.reshape(end-start, aug_time)

                sim_repre = F.cosine_similarity(ssl_repre.unsqueeze(dim=1), aug_repre.reshape(end-start, aug_time, -1), dim=2)
                # sim_repre = F.cosine_similarity(ssl_backbone_out.unsqueeze(dim=1), aug_backbone_out.reshape(end-start, aug_time, -1), dim=2)
                
                sim_aug = get_cos_similarity_3d(aug_repre.reshape(end-start, aug_time, -1), aug_repre.reshape(end-start, aug_time, -1))

                ssl_labels = torch.cat([ssl_labels, ssl_label], dim=0)
                aug_labels = torch.cat([aug_labels, aug_label], dim=0)
                ssl_repres = torch.cat([ssl_repres, ssl_repre], dim=0)
                aug_repres = torch.cat([aug_repres, aug_repre], dim=0)
                # ssl_repres = torch.cat([ssl_repres, ssl_backbone_out], dim=0)
                # aug_repres = torch.cat([aug_repres, aug_backbone_out], dim=0)
                sim_with_ori = torch.cat([sim_with_ori, sim_repre], dim=0)
                sim_representation = torch.cat([sim_representation, sim_aug], dim=0)

                # print((aug_repre - ssl_repre.unsqueeze(dim=1)).norm(dim=-1, p=2).mean())
                # print(sim.sum(dim=-1).float().mean())
                # print(f'For Iteration {index}, Mean---Similarity with Original Image: {sim.mean()}, Mean---Representation Similarity: {sim_repre.mean()}')
            
            labels = labels.to(device)
            # print(f'{attack} Augmented Images Similarity Mean, Mean: {sim_representation.mean()}')
            print(f'{attack}, Similarity with Ori Mean: {sim_with_ori.mean(dim=-1).mean()}, Variance: {sim_with_ori.var(-1).mean()}')
            print(f'{attack}, Target Model Label equals Aug Labels Count: {(labels.unsqueeze(dim=1) == aug_labels).sum(-1).float().mean()}')
            print(f'{attack}, SSL Model Label equals Aug Labels Count: {(ssl_labels.unsqueeze(dim=1) == aug_labels).sum(-1).float().mean()}')

            # auc_metric = detect_adv_by_label_rep(labels, aug_labels, sim_with_ori, aug_time)
            # auc_metric = detect_adv_by_rep_label(labels, aug_labels, sim_with_ori, aug_time)
            auc_metric = detect_adv_by_label_sim(labels, aug_labels, aug_time)
            # auc_metric = detect_adv_by_representation(labels, aug_labels, sim_with_ori)
            auc_metrics.append(auc_metric)

            # auc_metrics.append(sim_with_ori.sort(descending=True, dim=-1)[0].cpu().numpy())


            # end = time.time()
            # print(end)
            # print(f'Runing time: {end-start}')
        # # np.save('./auc_cifar10_normal_label.npy', auc_metrics)
        metrics = plot_roc(auc_metrics, attacks[1:], './auc_gray.png')
        # search_k(auc_metrics, attacks)

        # np.save('./auc_c10_lr.npy', metrics)





if __name__ == "__main__":

    main()
