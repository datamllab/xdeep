"""
Created on Wed Mar 28 10:12:13 2018
@author: Utku Ozbulak - github.com/utkuozbulak
"""

import torch


def generate_smooth_grad(explainer, input_img, target_class=None, param_n=10, param_sigma_multiplier=4):
    """
        Generates smooth gradients of given Backprop type. You can use this with both vanilla
        and guided backprop
    Args:
        Backprop (class): Backprop type
        prep_img (torch Variable): preprocessed image
        target_class (int): target class of imagenet
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """
    # Generate an empty image/matrix
    smooth_grad = torch.zeros(input_img.size()).squeeze()

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(input_img).sub(torch.min(input_img))).item()
    for x in range(param_n):
        # Generate noise
        noise = input_img.data.new(input_img.size()).normal_(mean, sigma**2)
        # Add noise to the image
        noisy_img = input_img + noise
        vanilla_grads = explainer.calculate_gradients(noisy_img, target_class)
        # Add gradients to smooth_grad
        smooth_grad = smooth_grad + vanilla_grads
    # Average it out
    smooth_grad = smooth_grad / param_n
    return smooth_grad