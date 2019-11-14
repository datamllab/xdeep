import torch


def generate_smooth_grad(explainer, input_, target_class=None, n=50, mean=0, sigma_multiplier=4):

    """ Generates smooth gradients of given explainer.

        # Arguments
            explainer: class. Backprop method.
            input_: torch.Tensor. Preprocessed image with shape (1, C, H, W).
            target_class: int. Index of target class. Defaults to None.
            n: int. Amount of noisy images used to smooth gradient. Default to 10.
            mean: int. Mean value of normal distribution when generating noise. Default to 0.
            sigma_multiplier: int. Sigma multiplier when calculating std of noise. Default to 4.

        # Return:
            Smooth gradient (torch.Tensor) with shape (C, H, W).
    """

    # Generate an empty image with shape (C, H, W) to save smooth gradient
    smooth_grad = torch.zeros(input_.size()).squeeze()

    sigma = sigma_multiplier / (torch.max(input_)-torch.min(input_)).item()

    x = 0
    while x < n:
        # Generate noise
        noise = input_.new(input_.size()).normal_(mean, sigma**2)

        # Add noise to the image
        noisy_img = input_ + noise

        # calculate gradient for noisy image
        grads = explainer.calculate_gradients(noisy_img, target_class)

        # accumulate gradient
        smooth_grad = smooth_grad + grads

        x += 1

    # average
    smooth_grad = smooth_grad / n

    return smooth_grad
