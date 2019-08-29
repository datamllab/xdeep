'''
Misa Ogura. (2019, July 8). MisaOgura/flashtorch: 0.0.8 (Version v0.0.8). Zenodo. http://doi.org/10.5281/zenodo.3271410
'''

import torch
import torch.nn as nn
import warnings


class Backprop:
    def __init__(self, model, guided=False):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.guided = guided
        self.handle = []
        self._register_conv_hook()

    def calculate_gradients(self,
                            input_,
                            target_class=None,
                            take_max=False,
                            use_gpu=False):

        if 'inception' in self.model.__class__.__name__.lower():
            if input_.size()[1:] != (3, 299, 299):
                raise ValueError('Image must be 299x299 for Inception models.')

        if self.guided:
            self.relu_outputs = []
            self._register_relu_hooks()

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

            # Create a 2D tensor with shape (1, num_classes) and
            # set all element to zero

            target = torch.FloatTensor(1, output.shape[-1]).zero_()

            if torch.cuda.is_available() and use_gpu:
                target = target.to('cuda')

            if (target_class is not None) and (top_class != target_class):
                warnings.warn(UserWarning(
                    f'The predicted class index {top_class.item()} does not' +
                    f'equal the target class index {target_class}. Calculating' +
                    'the gradient w.r.t. the predicted class.' ))

            # Set the element at top class index to be 1
            target[0][top_class] = 1

        # Calculate gradients of the target class output w.r.t. input_
        output.backward(gradient=target)

        # Detach the gradients from the graph and move to cpu
        gradients = self.gradients.detach().cpu()[0]

        if take_max:
            # Take the maximum across colour channels
            gradients = gradients.max(dim=0, keepdim=True)[0]

        for module in self.handle:
            module.remove()

        return gradients

    def _register_conv_hook(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and module.in_channels == 3:
                backward_handle = module.register_backward_hook(_record_gradients)
                self.handle.append(backward_handle)

    def _register_relu_hooks(self):
        def _record_output(module, input_, output):
            self.relu_outputs.append(output)

        def _clip_gradients(module, grad_in, grad_out):
            relu_output = self.relu_outputs.pop()
            clippled_grad_out = grad_out[0].clamp(0.0)

            return (clippled_grad_out.mul(relu_output),)

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                forward_handle = module.register_forward_hook(_record_output)
                backward_handle = module.register_backward_hook(_clip_gradients)
                self.handle.append(forward_handle)
                self.handle.append(backward_handle)