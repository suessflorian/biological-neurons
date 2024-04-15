import torch
import foolbox as fb

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

def foolbox_attack(model, images, labels, attack, epsilons):
    raise NotImplementedError('Function "foolbox_attack" is not finished.')
    model.eval()
    # Attack on a single batch
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    # try:
    raw_attack, perturbed_image, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
    with torch.no_grad():
        original_predictions = model(images).argmax(dim = -1)
        perturbed_predictions = model(perturbed_image).argmax(dim = -1)
        return raw_attack, perturbed_image, perturbed_predictions, original_predictions
    # except:
    #     return None, None, None, model(images).argmax(dim = -1)