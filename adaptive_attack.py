import os
import math
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torchvision.transforms.functional as TF

from resnet18_32x32 import ResNet18_32x32
from utils.resnet_factory import SimSiamWithCls


def tensor2variable(x=None, device=None, requires_grad=False):

    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)

def multi_transform(img, transforms, times=50):

    return torch.stack([transforms(img) for t in range(times)], dim=1)


def tar_predict(target_model, samples, labels, device=None):

    if type(samples) == np.ndarray:
        samples = torch.from_numpy(samples)
    else: 
        labels = labels.detach().cpu().numpy()
    samples = samples.to(device)
    pred_labels = target_model(samples.float())
    pred_labels = torch.max(pred_labels, 1)[1]
    pred_labels = pred_labels.cpu().numpy()

    acc = (pred_labels == labels.argmax(-1)).sum()/len(labels)
    return acc, pred_labels

def ssl_predict(ssl_backbone, ssl_classifier, samples, labels, device=None):

    if type(samples) == np.ndarray:
        samples = torch.from_numpy(samples)
    else: 
        labels = labels.cpu().detach().numpy()
    samples = samples.to(device)
    pred_labels = ssl_backbone(samples.float())
    pred_labels = ssl_classifier(pred_labels)
    pred_labels = torch.max(pred_labels, 1)[1]

    acc = (pred_labels.cpu().numpy() == labels.argmax(-1)).sum()/len(labels)
    return acc, pred_labels




def ada_attack(target_model, ssl_backbone, ssl_classifer, ssl_projector, criterion, X, y_true, tar_labels, img_transforms=None,
        alpha=-1.0, aug_time=50, epsilon=8/256, bound=(0,1), step_size=0.01, num_iter=50, randomize=False, logger=None):

    target_model.eval()
    ssl_backbone.eval()
    ssl_classifer.eval()
    ssl_projector.eval()

    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):

        adv_samples = X + delta
        trans_samples = multi_transform(adv_samples, img_transforms, aug_time)

        tar_loss = criterion(target_model(adv_samples), torch.max(tar_labels, 1)[1])
        tar_loss.backward()
        tar_grad_data = delta.grad.detach().sign()
        delta.grad.zero_()

        ori_img_transforms = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ssl_backbone_outs = ssl_backbone(ori_img_transforms(adv_samples))
        # ssl_backbone_outs, _ = ssl_backbone(adv_samples)
        ssl_ori_rep = ssl_projector(ssl_backbone_outs)
        ssl_ori_out = ssl_classifer(ssl_backbone_outs)

        aug_backbone_outs = ssl_backbone(trans_samples.reshape(-1,3,32,32))
        aug_reps = ssl_projector(aug_backbone_outs)
        aug_outs = ssl_classifer(aug_backbone_outs)
        ssl_cls_loss = criterion(aug_outs, torch.max(tar_labels,1)[1].repeat_interleave(aug_time))
        # ssl_cls_loss = criterion(ssl_ori_out, torch.max(tar_labels, 1)[1])

        ssl_rep_loss = F.cosine_similarity(ssl_ori_rep.unsqueeze(dim=1), aug_reps.reshape(len(adv_samples), aug_time, -1), dim=2).mean()
        ssl_loss = ssl_rep_loss * alpha + ssl_cls_loss

        ssl_loss.backward()
        ssl_grad_data = delta.grad.detach().sign()

        delta.data = (delta - step_size*(tar_grad_data+ssl_grad_data)).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        delta.grad.zero_()


    return (X + delta).clamp(*bound)


def batch_ada_attack(target_model, ssl_backbone, ssl_classifer, ssl_projector, criterion, samples, labels, tar_labels, img_transforms=None,
        aug_time=50, batch_size=100, alpha=-1.0, epsilon=8/256, bound=(0,1), step_size=0.002, num_iter=50, randomize=False, logger=None, device=None):
    

    assert len(samples) == len(labels)

    adv_samples = []
    number_batch = int(math.ceil(len(samples) / batch_size))

    print(f"Start Adaptive Attack, batch num: {number_batch}")
    for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(samples))
            # print(f'\r===> in batch {index:>2}, {end-start:>4} ({end:>4} in total) nature examples are perturbed ... ')
            batch_images = tensor2variable(torch.from_numpy(samples[start:end]), device, requires_grad=True)
            batch_labels = tensor2variable(torch.from_numpy(labels[start:end]).float(), device, requires_grad=True)
            batch_tar_labels = tensor2variable(torch.from_numpy(tar_labels[start:end]).float(), device, requires_grad=True)

            batch_adv_images = ada_attack(
                target_model, ssl_backbone, ssl_classifer, ssl_projector, 
                criterion, batch_images, batch_labels, batch_tar_labels, 
                img_transforms=img_transforms, alpha=alpha, aug_time=aug_time, epsilon=epsilon, bound=bound, step_size=step_size, num_iter=num_iter, randomize=randomize, logger=logger
                )

            adv_samples.extend(batch_adv_images.detach().cpu().numpy())

    return np.array(adv_samples)

def main(args):

    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nature_samples = np.load('./AEs/clean_inputs.npy')
    labels_samples = np.load('./AEs/clean_labels.npy')

    target_labels = np.roll(labels_samples, 1, 1)

    img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    # ssl model
    model = SimSiamWithCls()
    model.load_state_dict(torch.load(f"{args.weights_dir}/{args.ssl_model}.pth"))
    model.to(device)
    model.eval()

    backbone = model.backbone
    classifier = model.classifier
    projector = model.projector

    ssl_acc, _ = ssl_predict(backbone, classifier, nature_samples, labels_samples, device)


    # target model
    target_model = ResNet18_32x32()
    target_model.load_state_dict(torch.load(f"{args.weights_dir}/{args.model_name}.pth", map_location='cpu'))
    target_model.to(device)
    target_model.eval()

    tar_acc, _ = tar_predict(target_model, nature_samples, labels_samples, device)
    print(f"Model Accuracy: SSL model: {ssl_acc:.2f}, Target model: {tar_acc:.2f}")

    criterion = nn.CrossEntropyLoss()


    print("Start Attack_________________________")

    alpha = args.alpha
    step_size = args.step_size
    epsilon = args.e
    aug_num = args.aug_num
    batch_size = args.bs
    num_iter = args.num_iter

    adv_samples = batch_ada_attack(
        target_model, backbone, classifier, projector, criterion, 
        nature_samples, labels_samples, target_labels, img_transforms, step_size=step_size, 
        aug_time=aug_num, batch_size=batch_size, alpha=alpha, epsilon=epsilon, num_iter=num_iter, device=device
    )

    ssl_acc, ssl_adv_labels = ssl_predict(backbone, classifier, adv_samples, labels_samples, device=device)
    tar_acc, tar_adv_labels = tar_predict(target_model, adv_samples, labels_samples, device=device)

    print(f"Attack Success Rate: SSL model: {1-ssl_acc:.2f} Target model: {1-tar_acc:.2f}")
    
    np.save(f'./AEs/adaptive/Ada_AdvSamples.npy', adv_samples)
    np.save(f'./AEs/adaptive/Ada_AdvLabels.npy', tar_adv_labels)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Adaptive Attacks')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--weights_dir', type=str, default='./weights', help='the directory to store model weights')
    parser.add_argument('--model_name', type=str, default='resnet_c10')
    parser.add_argument('--ssl_model', type=str, default='simsiam_c10')

    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--e', type=float, default=8/255., help='perturbation budget')
    parser.add_argument('--step_size', type=float, default=0.002, help='step size in PGD')
    parser.add_argument('--num_iter', type=int, default=50, help='iteration number in PGD')
    parser.add_argument('--aug_num', type=int, default=50, help='number of augmentation')
    parser.add_argument('--alpha', type=float, default=-1)




    arguments = parser.parse_args()
    main(arguments)