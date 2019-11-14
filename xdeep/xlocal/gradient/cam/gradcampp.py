import torch
import torch.nn.functional as F
from .gradcam import GradCAM


class GradCAMpp(GradCAM):
    """
        GradCAM++, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super(GradCAMpp, self).__init__(model_dict)

    def forward(self, input_image, class_idx=None, retain_graph=False):

        """Generates GradCAM++ result.

        # Arguments
            input_image: torch.Tensor. Preprocessed image with shape (1, C, H, W).
            class_idx: int. Index of target class. Defaults to be index of predicted class.

        # Return
            Result of GradCAM++ (torch.Tensor) with shape (1, H, W).
        """

        b, c, h, w = input_image.size()

        logit = self.model_arch(input_image)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)

        global_sum = activations.view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = gradients.pow(2).mul(2) + global_sum.mul(gradients.pow(3))

        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp() * gradients)
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map