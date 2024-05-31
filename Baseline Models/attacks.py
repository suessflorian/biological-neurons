import torch
import foolbox as fb
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

def foolbox_attack(model, images, labels, attack, epsilons, device):
    # raise NotImplementedError('Function "foolbox_attack" is not finished.')
    model.eval()
    # Attack on a single batch
    fmodel = fb.PyTorchModel(model, bounds=(0,1), device=device)
    # try:
    raw_attack, perturbed_image, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
    with torch.no_grad():
        original_predictions = model(images).argmax(dim = -1)
        perturbed_predictions = model(perturbed_image).argmax(dim = -1)
        return raw_attack, perturbed_image, perturbed_predictions, original_predictions
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
    

