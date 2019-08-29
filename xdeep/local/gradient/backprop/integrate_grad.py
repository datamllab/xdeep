"""
Borrow some code from https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""

import torch
import numpy as np


def generate_integrated_grad(explainer, input_image, target_class=None, steps=5):
    """
            Generates integrate gradients of given Backprop type. You can use this with both vanilla
            and guided backprop
        Args:
            explainer (class): Backprop type
            input_img (torch Variable): preprocessed image
            target_class (int): target class of imagenet
            steps (int): times of integrate
        """

    # Generate xbar images
    step_list = np.arange(steps + 1) / steps
    integrated_grads = torch.zeros(input_image.size()).squeeze()

    for step in step_list[1:]:
        single_integrated_grad = explainer.calculate_gradients(input_image*step, target_class)
        integrated_grads = integrated_grads + single_integrated_grad / steps
    return integrated_grads