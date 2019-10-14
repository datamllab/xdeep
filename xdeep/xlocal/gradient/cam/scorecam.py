import torch
import torch.nn.functional as F
from .basecam import *


class ScoreCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):

        """Generates ScoreCAM result.

        # Arguments
            input_: torch.Tensor. Preprocessed image with shape (1, C, H, W).
            class_idx: int. Index of target class. Defaults to be index of predicted class.

        # Return
            Result of GradCAM (torch.Tensor) with shape (1, H, W).
        """

        b, c, h, w = input.size()

        logit = self.model_arch(input)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, predicted_class].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, predicted_class].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        score_saliency_map = torch.zeros((1, 1, h, w))

        with torch.no_grad():
            for i in range(k):
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.relu(saliency_map)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if (saliency_map.max() == saliency_map.min()).numpy().tolist() == 1:
                    continue

                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                output = self.model_arch(input * norm_saliency_map)
                output = (output - output.mean()) / output.std()
                score = output[0][predicted_class]
                score = F.softmax(score)
                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
        return score_saliency_map
