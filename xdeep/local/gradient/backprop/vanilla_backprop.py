'''
Misa Ogura. (2019, July 8). MisaOgura/flashtorch: 0.0.8 (Version v0.0.8). Zenodo. http://doi.org/10.5281/zenodo.3271410
'''

import warnings

import torch
import torch.nn as nn

class Backprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        #self._remove_former_hook()
        self._register_conv_hook()

    def calculate_gradients(self,
                            input_,
                            target_class=None,
                            take_max=False,
                            use_gpu=False):

        if 'inception' in self.model.__class__.__name__.lower():
            if input_.size()[1:] != (3, 299, 299):
                raise ValueError('Image must be 299x299 for Inception models.')

        if torch.cuda.is_available() and use_gpu:
            self.model = self.model.to('cuda')
            input_ = input_.to('cuda')

        self.model.zero_grad()

        self.gradients = torch.zeros(input_.shape)

        output = self.model(input_)

        if len(output.shape) == 1:
            target = None
        else:
            _, top_class = output.topk(1, dim=1)

            target = torch.FloatTensor(1, output.shape[-1]).zero_()

            if torch.cuda.is_available() and use_gpu:
                target = target.to('cuda')

            if (target_class is not None) and (top_class != target_class):
                warnings.warn(UserWarning(
                    f'The predicted class index {top_class.item()} does not' +
                    f'equal the target class index {target_class}. Calculating' +
                    'the gradient w.r.t. the predicted class.'
                ))

            # Set the element at top class index to be 1

            target[0][top_class] = 1

        # Calculate gradients of the target class output w.r.t. input_

        output.backward(gradient=target)

        # Detach the gradients from the graph and move to cpu

        gradients = self.gradients.detach().cpu()[0]

        if take_max:
            # Take the maximum across colour channels
            gradients = gradients.max(dim=0, keepdim=True)[0]

        return gradients

    def _register_conv_hook(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and \
                    module.in_channels == 3:
                module.register_backward_hook(_record_gradients)

    #def _remove_former_hook(self):
    #    for _, module in self.model.named_modules():
    #            module.remove()

'''
import torchvision.models as models

from utils import ImageNetIndex
from utils import (apply_transforms, load_image, visualize)

# 读取图片为Image格式
image = load_image('./images/great_grey_owl.jpg')
# 对图像进行Resize、Normalize、Totensor操作
input_ = apply_transforms(image)

# 获取target class
imagenet = ImageNetIndex()
target_class = imagenet['great grey owl']

# 定义模型
model = models.alexnet(pretrained=True)

# 定义解释器
backprop = Backprop(model)

# 得到解释器结果
gradients = backprop.calculate_gradients(input_, target_class)
max_gradients = backprop.calculate_gradients(input_, target_class, take_max=True)

# 可视化
visualize(input_, gradients, save_path='/Users/apple/Desktop/vanilla_bp.jpg')
'''