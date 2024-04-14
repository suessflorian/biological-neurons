import torch


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