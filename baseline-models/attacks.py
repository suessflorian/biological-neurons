import torch
import foolbox as fb
import numpy as np
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

import copy
from art.attacks.evasion import FastGradientMethod, SquareAttack
from art.estimators.classification import PyTorchClassifier
from skimage.metrics import structural_similarity as ssim

def fgsm_attack(model, loss_fn, images, labels, epsilon):
    '''Fast Gradient Sign Method'''
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    image_gradients = images.grad.data
    perturbed_image = images + epsilon * image_gradients.sign()
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    output = model(perturbed_image)
    return perturbed_image, output.argmax(dim = -1)

def foolbox_attack(model, images, labels, attack, epsilons, device, verbose = True):
    # raise NotImplementedError('Function "foolbox_attack" is not finished.')
    model.eval()
    # Attack on a single batch
    fmodel = fb.PyTorchModel(model, bounds=(0,1), device=device)
    raw_attack, perturbed_image, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
    with torch.no_grad():
        original_predictions = model(images).argmax(dim = -1)
        perturbed_predictions = model(perturbed_image).argmax(dim = -1)
        return raw_attack, perturbed_image, perturbed_predictions, original_predictions
    

def art_attack(model, images, labels, attack, epsilons, device, verbose = False):

    images, labels = images.to(device), labels.to(device)
    
    model = model.to(device)
    
    if 'SimBA' in str(attack):
        model = nn.Sequential(model, nn.Softmax(dim=1)).to(device)
    
    artIntermediaryClassifier = PyTorchClassifier(
        model = model,
        loss = nn.CrossEntropyLoss(), # this doesn't matter
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01), # this doesn't matter either
        input_shape= images.shape[1:],
        nb_classes= 10,
        clip_values=(-1,1)
    )
    
    if "SquareAttack" in str(attack):
        method = attack(estimator=artIntermediaryClassifier, eps=epsilons, verbose=verbose, batch_size=images.shape[0])
    elif 'SimBA' in str(attack):
        method = attack(classifier=artIntermediaryClassifier, epsilon=epsilons, verbose=verbose)
    elif "BoundaryAttack" in str(attack):
        method = attack(estimator=artIntermediaryClassifier, epsilon=epsilons, targeted=False, max_iter=500, verbose=verbose)
    
    # Non-epsilon based attacks
    elif "hopskipjump" in str(attack): # hopskipjump is NOT a typo
        print('WARNING: THE HOP SKIP JUMP ATTACK IS VERY SLOW')
        print('WARNING: This attack is NOT epsilon-based')
        method = attack(classifier=artIntermediaryClassifier, verbose=verbose)
    elif "ZooAttack" in str(attack):
        print('WARNING: ZOO ATTACK IS VERY SLOW')
        print('WARNING: This attack is NOT epsilon-based')
        method = attack(classifier=artIntermediaryClassifier, verbose=verbose)
    
    # if images.shape[-1] == 864:
    #     images = images.view(images.shape[0], 6, 12, 12)
    # elif images.shape[-1] == 256:
    #     images = images.view(images.shape[0], 16, 4, 4)
    # elif images.shape[-1] == 120:
    #     images = images.view(images.shape[0], 1, 10, 12)
    # elif images.shape[-1] == 84:
    #     images = images.view(images.shape[0], 1, 7, 12)
    # elif images.shape[-1] == 10:
    #     images = images.view(images.shape[0], 1, 2, 5)
    
    adv = method.generate(x=images.cpu().numpy())
    
    perturbed_images = torch.tensor(adv).to(device)
    raw_attack = torch.clamp(perturbed_images - images, min=-epsilons, max=epsilons)
    perturbed_images = images + raw_attack
    
    with torch.no_grad():
        model.eval()
        original_predictions = model(images).argmax(dim=-1)
        perturbed_predictions = model(perturbed_images).argmax(dim=-1)
        return raw_attack, perturbed_images, perturbed_predictions, original_predictions
    







# Black-box attacks: https://github.com/I-am-Bot/Black-Box-Attacks




def binsearch_boundary(src_pt,
                      dest_pt,
                      threshold,
                      model, target_label,
                      device=torch.device('cpu')
                     ) -> np.array:
    '''
    Find a point between two points that will lies on the boundary.
    :param src_pt:    point at which phi=0
    :param dest_pt:   point at which phi=1
    :param threshold: gap between source point and destination point
    '''

    while torch.linalg.norm(dest_pt - src_pt) >= threshold:
        midpoint = (src_pt + dest_pt) / 2
        if indicator_function(model, midpoint, target_label) == 1:
            dest_pt = midpoint
        else:
            src_pt = midpoint
    return dest_pt

def indicator_function(model, x, target_label):
    if torch.argmax(model(x)) == target_label:
        return 1
    else:
        return 0

def estimate_gradient(orig_pt,
                      step_size,
                      sample_count,
                      model, 
                      target_label,
                      device=torch.device('cpu')
                     ) -> np.ndarray:
    '''
    Estimate the gradient via Monte Carlo sampling.
    :param orig_pt:      point to estimate gradient at
    :param step_size:    length of each step in the proposed direction
    :param sample_count: number of Monte Carlo samples
    '''
    # sample directions

    directions = torch.from_numpy(np.random.randn(sample_count, orig_pt.shape[1], orig_pt.shape[2], orig_pt.shape[3])).float().to(device)
    directions /= torch.linalg.norm(directions, dim = (1, 2, 3), dtype = torch.float).reshape(sample_count, 1, 1, 1).repeat(1, orig_pt.shape[1], orig_pt.shape[2], orig_pt.shape[3])

    # get phi values
    values = torch.from_numpy(np.empty((sample_count, 1), dtype = np.float)).to(device)

    for i in range(sample_count):
        values[i, 0] = indicator_function(model, orig_pt + directions[i, :] * step_size, target_label) * 2 - 1
    # subtract from the mean

    avg = torch.sum(directions * values.reshape(sample_count, 1, 1 ,1), dim = 0) / (sample_count - 1)
    # project them to unit L2
    norm_avg = avg / torch.linalg.norm(avg)
    return norm_avg.float()

def gradient_descent(orig_pt,
                     grad,
                     step_size,
                     model, target_label
                    ) -> np.ndarray:
    '''
    Do gradient descent on a point already on the boundary.
    :param orig_pt:    point to do gradient descent on
    :param grad:       the estimated gradient
    :param step_size:  initial step size to try
    '''
    # find the step size to stay in phi=1

    while True:
        new_vector = orig_pt + step_size * grad
        if indicator_function(model, new_vector, target_label):
            break
        step_size /= 2
    return new_vector

def hopskipjumpattack(orig_pt,
                      model,
                      max_iter = 100,
                      init_grad_queries = 100,
                      binsearch_threshold = 2e-6,
                      dest_pt = None,
                      target_label = None,
                      device = torch.device('cpu')
                     ):
    '''
    Implementation of the HopSkipJumpAttack.
    :param orig_pt:             point at which phi=0
    :param max_iter:            (Optional) maximum number of optimization iteration.
                                Default is 100.
    :param init_grad_queries:   (Optional) initial query count to estimate gradient
                                Default is 100.
    :param binsearch_threshold: (Optional) the threshold to stop binary searching the boundary.
                                Default is 1e-6.
    :param dest_pt:             (Optional) point which phi=1.
                                If dest_pt is None, will be initialized to be a random vector.
                                For cases when one restarts this iterative algo. Default is 0.
    '''
    d = orig_pt.shape[0]
    # initialize a vector with phi=1
    if dest_pt is None:
        while True:
            dest_pt = torch.from_numpy(np.random.random_sample(d)).to(device)
            if indicator_function(model, dest_pt, target_label) == 1:
                break

    for it in range(1, max_iter + 1):

        print(f'Iter {it:03d}: ', end='')

        # project on to boundary
        boundary = binsearch_boundary(orig_pt, dest_pt, binsearch_threshold, model, target_label)
        # if the error is too small, return as is
        distance = torch.linalg.norm(boundary - orig_pt)
        if distance < binsearch_threshold:
            print('Step size too small, terminating...')
            # this works because we return the phi=1 endpoint in binsearch.
            return boundary

        # estimate the gradient
        step_size = torch.linalg.norm(dest_pt - orig_pt) / d
        sample_count = int(init_grad_queries * it ** 0.5)
        grad = estimate_gradient(boundary, step_size, sample_count, model, target_label)

        # and gradient descend
        step_size = torch.linalg.norm(boundary - orig_pt) / it ** 0.5
        dest_pt = gradient_descent(boundary, grad, step_size, model, target_label)
        distance = torch.linalg.norm(dest_pt - orig_pt)
        print(distance)

    return dest_pt

def hsja_attack(model, images, labels, loader, device):
    adv_images, target_labels = [], []
    
    for i in range(len(labels)):
        image = images[i].unsqueeze(0)
        label = labels[i]
        
        print(image.shape)
        print(label)
        
        ori_img, ori_label = image, label
        iterator = iter(loader)
        target_imgs, target_labels = next(iterator)
        target_img, target_label = target_imgs[0], target_labels[0]

        ori_img, ori_label = ori_img.to(device), ori_label.to(device)
        target_img, target_label = target_img.to(device), target_label.to(device)


        while (ori_label == target_label or torch.argmax(model(target_img)) != target_label):
            target_imgs, target_labels = next(iterator)
            target_img, target_label = target_imgs[0], target_labels[0]
            ori_img, ori_label = ori_img.to(device), ori_label.to(device)
            target_img, target_label = target_img.to(device), target_label.to(device)

        adv_img = hopskipjumpattack(ori_img, model, dest_pt = target_img, target_label = target_label)
        
        adv_images.append(adv_img)
        target_label.append(target_label)
        # print('original label:', ori_label)
        # print('target label:', target_label)
        # print(torch.argmax(model(target_img)) == target_label)
        # print(indicator_function(model, target_img, target_label))
        # print('attack label:', torch.argmax(model(adv_img)))
    return torch.cat(adv_images, dim=0), torch.cat(target_label, dim=0)
    # except:
    #     return None, None, None, model(images).argmax(dim = -1)

def art_attack(model, criterion, optimizer, input_shape, classes, clips, images, iterations, epsilons, device):
    # raise NotImplementedError('Function "foolbox_attack" is not finished.')
    model.eval()
    # Attack on a single batch
    artIntermediaryClassifier = PyTorchClassifier(
        model = model,
        loss = criterion,
        optimizer = optimizer,
        input_shape= input_shape,
        nb_classes= classes,
        clip_values=clips
    )
    # try:
    method = SquareAttack(estimator=artIntermediaryClassifier, max_iter=iterations, eps=epsilons, verbose=False)
    images = images.cpu().numpy()
    adv = method.generate(x=images)
    adv = torch.tensor(adv).to(device)
    images = torch.tensor(images).to(device)
    
    with torch.no_grad():
        original_predictions = model(images).argmax(dim = -1)
        perturbed_predictions = model(adv).argmax(dim = -1)
        return adv, perturbed_predictions, original_predictions
    
def find_clip_values(loader):
    min_val, max_val = float('inf'), float('-inf')
    for images, _ in loader:
        batch_min = torch.min(images)
        batch_max = torch.max(images)

        if batch_min < min_val:
            min_val = batch_min.item()
        if batch_max > max_val:
            max_val = batch_max.item()

    return min_val, max_val
    

