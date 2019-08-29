import torch
import torch.nn.functional as F
from ..utils.utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, \
    find_squeezenet_layer


class ScoreCAM(object):

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']

        self.model_arch = model_dict['arch']
        self.model_arch.eval()
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()  # (1,3,224,224)

        logit = self.model_arch(input)  # (1,1000)

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

        # 初始化saliency map
        score_saliency_map = torch.zeros((1, 1, h, w))

        for i in range(k):
            # 将第i张activation map上采样到（224，224)
            saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

            # 得到上采样后的activation map的二值图
            seg = saliency_map > saliency_map.min()
            seg = seg.float()

            # 上采样的第i张显著性二值化图*input，进行一次前向传播
            output = self.model_arch(input * seg)

            # output进行正则化
            output = (output - output.mean()) / output.std()

            # 得到target class上的预测分数
            score = output[0][predicted_class]

            # 将每一张显著性图加权相加，权值为target class上的置信分数，代表该显著性图对于target class的重要性
            score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)