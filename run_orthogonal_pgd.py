import os 
import math
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils.resnet_factory import ResNet18, SimSiamWithCls
from orthogonal_pgd import PGD

def multi_transform(img, transforms, times=50):

    return torch.stack([transforms(img) for t in range(times)], dim=1)


class Detector(nn.Module):

    def __init__(self, ssl_model = None, target_model=None, augmentation=None, aug_time=50, batch_size=100, device=None):
        
        super(Detector, self).__init__()
        self.target_model = target_model
        self.model = ssl_model
        self.model.eval()
        self.augmentation = augmentation
        self.aug_time = aug_time
        self.batch_size = batch_size
        self.device = device
    
    def forward(self, samples, clean_labels=None):

        device = self.device
        target_model = self.target_model.to(device)
        backbone = self.model.backbone.to(device)
        classifier = self.model.classifier.to(device)
        projector = self.model.projector.to(device)

        target_model.eval()
        backbone.eval()
        classifier.eval()
        projector.eval()

        preds = target_model(samples.to(device))
        labels = preds.max(-1)[1]
        labels = labels.cpu()

        number_batch = int(math.ceil(len(samples) / self.batch_size))
        sim_with_ori = torch.Tensor()

        ssl_repres = torch.Tensor()
        aug_repres = torch.Tensor()
        ssl_labels = torch.Tensor()
        aug_labels = torch.Tensor()

        for index in range(number_batch):
            start = index * self.batch_size
            end = min((index + 1) * self.batch_size, len(samples))

            batch_samples = samples[start:end].to(device)
            trans_images = multi_transform(batch_samples, self.augmentation, times=self.aug_time).to(device)

            normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ssl_backbone_out = backbone(normalization(batch_samples))

            ssl_repre = projector(ssl_backbone_out)
            ssl_label = classifier(ssl_backbone_out)

            aug_backbone_out = backbone(trans_images.reshape(-1, 3, 32, 32))
            aug_repre = projector(aug_backbone_out)
            aug_label = classifier(aug_backbone_out)
            aug_label = aug_label.reshape(end-start, self.aug_time, -1)

            sim_repre = F.cosine_similarity(ssl_repre.unsqueeze(dim=1), aug_repre.reshape(end-start, self.aug_time, -1), dim=2)

            ssl_labels = torch.cat([ssl_labels, ssl_label.cpu()], dim=0)
            aug_labels = torch.cat([aug_labels, aug_label.cpu()], dim=0)
            ssl_repres = torch.cat([ssl_repres, ssl_repre.cpu()], dim=0)
            aug_repres = torch.cat([aug_repres, aug_repre.cpu()], dim=0)
            sim_with_ori = torch.cat([sim_with_ori, sim_repre.cpu()], dim=0)
        
        return ssl_labels, aug_labels, sim_with_ori

def run_experiment(taget_model, detector, samples, labels, batch_size=30, device=None, mode='select', **attack_args):
    
    pgd = PGD(taget_model, detector, device=device, **attack_args)

    advx = pgd.attack(samples.clone(), labels, batch_size)

    if 'target' in attack_args and attack_args['target'] is not None:
        attack_succeeded = (taget_model(advx.to(device)).argmax(1).cpu()==attack_args['target'])
    else:
        attack_succeeded = (taget_model(advx.to(device)).argmax(1).cpu()!=labels)

    sr = torch.mean(attack_succeeded.float())
    print(f"attack success rate: {sr}")
    
    return advx

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

    samples = torch.tensor(np.load('./AEs/clean_inputs.npy'), dtype=torch.float32)
    labels = torch.tensor(np.load('./AEs/clean_labels.npy'), dtype=torch.int64)
    target_labels = torch.roll(labels, 1, 1)

    labels = labels.max(-1)[1]
    target_labels = target_labels.max(-1)[1]

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

    target_model = ResNet18()
    target_model.load_state_dict(torch.load(f"{args.weights_dir}/{args.model_name}.pth", map_location='cpu'))
    target_model.to(device)
    target_model.eval()

    ssl_model = SimSiamWithCls()
    ssl_model.load_state_dict(torch.load(f"{args.weights_dir}/{args.ssl_model}.pth"))
    ssl_model.to(device)
    ssl_model.eval()

    epsilon = args.e
    aug_num = args.aug_num
    batch_size = args.bs
    steps = args.steps
    alpha = args.alpha
    mode = args.mode
    target = args.target
    n_ae = len(samples) # number of AEs

    detector = Detector(ssl_model, target_model, augmentation=img_transforms, aug_time=aug_num, batch_size=batch_size, device=device)

    if mode == 'select':
        d = {'use_projection': True, 'eps': epsilon, 'alpha': alpha, 'steps': steps,
            'projection_norm': 'linf'
        }
    
    else: 
        d = {'use_projection': True, 'eps': epsilon, 'alpha': alpha, 'steps': steps,
            'projection_norm': 'linf', 'project_detector': True, 'project_classifier': True
        }


    if target:
        advx = run_experiment(target_model, detector, samples[:n_ae], target_labels[:n_ae], batch_size=64,
                            device=device, mode=mode,
                            classifier_loss=nn.CrossEntropyLoss(),
                            detector_loss=None,
                            target=target_labels[:n_ae],
                            **d)
    else:
        advx = run_experiment(target_model, detector, samples[:n_ae], labels[:n_ae], batch_size=64,
                            device=device, mode=mode,
                            classifier_loss=nn.CrossEntropyLoss(),
                            detector_loss=None,
                            target=None,
                            **d)
        
    np.save(f'./AEs/adaptive/Opgd_AdvSamples.npy', advx.detach().cpu().numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Orthogonal PGD')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--weights_dir', type=str, default='./weights', help='the directory to store model weights')
    parser.add_argument('--model_name', type=str, default='resnet_c10')
    parser.add_argument('--ssl_model', type=str, default='simsiam_c10')

    parser.add_argument('--mode', type=str, help='attack mode: select, orthogonal')

    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--e', type=float, default=8/255., help='perturbation budget')
    parser.add_argument('--steps', type=int, default=1000, help='iteration number')
    parser.add_argument('--aug_num', type=int, default=50, help='number of augmentation')
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.001)


    arguments = parser.parse_args()
    main(arguments)
