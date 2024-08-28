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

from resnet18_32x32 import ResNet18_32x32


def tensor2variable(x=None, device=None, requires_grad=False):

    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


def fgsm(model, criterion, X, y=None, device=None, epsilon=0.1, bound=(0,1)):
    """ 
    Construct FGSM adversarial examples on the examples X
    input: np array
    output: np array
    """

    X = tensor2variable(torch.from_numpy(X), device, requires_grad=True)
    y = tensor2variable(torch.from_numpy(y).float(), device, requires_grad=True)

    model.eval()
    delta = torch.zeros_like(X, requires_grad=True)

    loss = criterion(model(X + delta), y.max(-1)[1])
    loss.backward()
    delta = epsilon * delta.grad.detach().sign()

    return (X + delta).clamp(*bound).detach().cpu().numpy()



def pgd_linf(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), step_size=0.002, num_iter=50, randomize=False, device=None):
    """ 
    Construct PGD adversarial examples on the examples X
    input: np array
    output: np array
    """
    X = tensor2variable(torch.from_numpy(X), device, requires_grad=True)
    y = tensor2variable(torch.from_numpy(y).float(), device, requires_grad=True)

    model.eval()
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):

        loss = criterion(model(X + delta), y.max(-1)[1])
        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        delta.grad.zero_()

    return (X + delta).clamp(*bound).detach().cpu().numpy()

def cw_batch_attack(model, images, labels, batch_size, device=None):
    """
    input: numpy array
    output: numpy array
    """

    assert len(images) == len(labels)

    adv_sample = []
    number_batch = int(math.ceil(len(images) / batch_size))
    for index in range(number_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, len(images))
        print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
        batch_images = tensor2variable(torch.from_numpy(images[start:end]), device, requires_grad=True)
        batch_labels = tensor2variable(torch.from_numpy(labels[start:end]).float(), device, requires_grad=True)

        batch_adv_images = cw_attack(model, batch_images, batch_labels)
        adv_sample.extend(batch_adv_images.detach().cpu().numpy())
    return np.array(adv_sample)


def cw_attack(model, images, labels, targeted=False, c=10, kappa=0, max_iter=1000, learning_rate=0.01) :


    # b_min = 0                                
    # b_max = 1
    # b_mul=(b_max-b_min)/2.0
    # b_plus=(b_min+b_max)/2.0
    adv_images = images

    # Define f-function
    def f(x):
        
        outputs = model(x)


        i, _ = torch.max((1-labels)*outputs, dim=1)
        j = torch.masked_select(outputs, labels.bool())
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    # w = torch.zeros_like(images, requires_grad=True).to(device)
    w = torch.arctan(2*images-1).cuda()
    w_pert = torch.zeros_like(images, requires_grad=True).cuda()

    optimizer = optim.Adam([w_pert], lr=learning_rate)

    prev = 1e10
    for iter_index in range(1,max_iter+1):

            optimizer.zero_grad()
            adv_img = 1/2*(nn.Tanh()(w+w_pert) + 1)

            loss1 = nn.MSELoss(reduction='sum')(adv_img, images)
            loss2 = torch.sum(c*f(adv_img))
            loss = loss1 + loss2
            loss.backward(retain_graph=True)
            optimizer.step()


            if iter_index%200==0:
                 print(f'Iters: [{iter_index}/{max_iter}]\tLoss: {loss},Loss1(L2 distance):{loss1}, Loss2:{loss2}')
            
            if iter_index % (max_iter//10) == 0 :
                if loss > prev :
                    print('Attack Stopped due to CONVERGENCE....')
                    return adv_img
                prev = loss

    adv_images = 1/2*(nn.Tanh()(w+w_pert) + 1)
    return adv_images


def predict(model=None, samples=None, device=None):
    """

    :param model:
    :param samples:
    :param device:
    :return:
    """
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        samples = torch.tensor(samples).to(device)
        predictions = model(samples.float())
    return predictions


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
    model = ResNet18_32x32()
    weights = torch.load(f"{args.weights_dir}/{args.model_name}.pth", map_location='cpu')
    model.load_state_dict(weights)
    model.to(device)

    model.eval()

    nature_samples = np.load('./AEs/clean_inputs.npy')
    labels_samples = np.load('./AEs/clean_labels.npy')

    pred_labels = predict(model, nature_samples, device)
    pred_labels = torch.max(pred_labels, 1)[1]
    pred_labels = pred_labels.cpu().numpy()

    print(f"Model Accuracy is {((pred_labels == labels_samples.argmax(-1)).sum()/len(labels_samples))*100:.2f}")

    # Generate AEs
    attack_name = args.attack.upper() # fgsm, pgd, cw

    criterion = nn.CrossEntropyLoss()

    if attack_name == "FGSM":
        adv_samples = fgsm(model, criterion, nature_samples, labels_samples, device=device, epsilon=args.e)
    elif attack_name == "PGD":
        adv_samples = pgd_linf(model, criterion, nature_samples, labels_samples, device=device, epsilon=args.e, step_size=args.step_size, num_iter=args.num_iter)
    elif attack_name == "CW":
        adv_samples = cw_batch_attack(model, nature_samples, labels_samples, batch_size=args.bs, device=device)
    else:
        raise Exception("Unknown Attack")
    
    adv_labels = predict(model, adv_samples, device)
    adv_labels = torch.max(adv_labels, 1)[1]
    adv_labels = adv_labels.cpu().numpy()

    mis = (labels_samples.argmax(-1) != adv_labels).sum()

    print(f"{attack_name.upper()}, Attack success rate is {mis / len(adv_labels) * 100:.2f}")

    np.save(f'./AEs/raw/{attack_name}_AdvSamples.npy', adv_samples)
    np.save(f'./AEs/raw/{attack_name}_AdvLabels.npy', adv_labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Attacks: FGSM, PGD, CW')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--weights_dir', type=str, default='./weights', help='the directory to store model weights')
    parser.add_argument('--model_name', type=str, default='resnet_c10')

    parser.add_argument('--attack', type=str, help='attack methods: fgsm, pgd, cw')

    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--e', type=float, default=8/255., help='perturbation budget')
    parser.add_argument('--step_size', type=float, default=0.002, help='step size in PGD')
    parser.add_argument('--num_iter', type=int, default=50, help='iteration number in PGD')


    arguments = parser.parse_args()
    main(arguments)

    # python attacks.py --attack=fgsm --e=0.05
    # python attacks.py --attack=pgd
    # python attacks.py --attack=cw --bs=1000

    



