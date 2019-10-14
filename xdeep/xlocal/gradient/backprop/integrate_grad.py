import torch
import numpy as np


def generate_integrated_grad(explainer, input_, target_class=None, n=25):

    """ Generates integrate gradients of given explainer. You can use this with both vanilla
        and guided backprop

        # Arguments:
            explainer: class. Backprop method.
            input_: torch.Tensor. Preprocessed image with shape (N, C, H, W).
            target_class: int. Index of target class. Default to None.
            n: int. Integrate steps. Default to 10.

        # Return:
            Integrated gradient (torch.Tensor) with shape (C, H, W).
    """

    # Generate xbar images
    step_list = np.arange(n + 1) / n

    # Remove zero
    step_list = step_list[1:]

    # Create a empty tensor
    integrated_grads = torch.zeros(input_.size()).squeeze()

    for step in step_list:
        single_integrated_grad = explainer.calculate_gradients(input_*step, target_class)
        integrated_grads = integrated_grads + single_integrated_grad / n

    return integrated_grads
