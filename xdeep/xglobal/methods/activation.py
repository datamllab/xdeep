'''
This module is built on flashtorch. (Misa Ogura (2019, September 26).
                                    MisaOgura/flashtorch: 0.1.1 (Version v0.1.1).
                                    Zenodo. http://doi.org/10.5281/zenodo.3461737)

Besides filter visualization and deepdream, we also support layer visualization and logit visualization.
'''


import torch
import torch.nn as nn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random
import warnings
import numpy as np
from xdeep.utils import (load_image, apply_transforms, format_for_plotting, standardize_and_clip)


class GradientAscent(object):
    """
        Provides an interface for activation maximization via gradient ascent.
    """

    def __init__(self, model, img_size=224, lr=0.1, use_gpu=False):

        self.model = model
        self._img_size = img_size
        self._lr = lr
        self._use_gpu = use_gpu

        self.activation = None
        self.gradients = None

        self.handlers = []

        self.output = None

    def _filter_register_forward_hooks(self, layer, filter_idx):

        """
            Save forward propagation output on target filter.
        """

        def _record_activation(module, input_, output):
            self.activation = torch.mean(output[:,filter_idx,:,:])
        return layer.register_forward_hook(_record_activation)

    def _layer_register_forward_hooks(self, layer):
        """
            Save forward propagation output on target layer.
        """
        def _record_activation(module, input_, output):
            self.activation = torch.mean(output)
        return layer.register_forward_hook(_record_activation)

    def _logit_register_forward_hooks(self, layer, filter_idx):
        """
            Save forward propagation output on target logit.
        """
        def _record_activation(module, input_, output):
            self.activation = torch.mean(output[:, filter_idx])
        return layer.register_forward_hook(_record_activation)

    def _register_backward_hooks(self):
        """
            Save backward propagation gradient w.r.t input.
        """
        def _record_gradients(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and module.in_channels == 3:
                return module.register_backward_hook(_record_gradients)

    def _validate_filter_idx(self, layer_num_filters, filter_idx):
        """
            validate type and error of filter index
        """
        if not np.issubdtype(type(filter_idx), np.integer):
            raise TypeError('Index must be integer.')
        elif (filter_idx < 0) or (filter_idx >= layer_num_filters):
            raise ValueError(f'Filter index must be between 0 and {layer_num_filters - 1}.')

    def optimize(self, layer, filter_idx=None, input_=None, num_iter=30):
        """

        # Arguments:
            layer: torch.nn.modules. Target layer. Currently support for Conv.2D and Linear layer.
            filter_idx: int or list. The index of the target filter.
            input_: torch.Tensor. Optimized instance. Default to None.
            num_iter: int. The number of iteration for gradient ascent. Default to 30.

        # Returns:
            output (list of torch.Tensor): Optimized result at each iteration. With shape (num_iter, C, H, W).
        """

        # input_ has to be specified in deepdream
        if input_ is None:
            input_ = np.uint8(np.random.uniform(150, 180, (self._img_size, self._img_size, 3)))
            input_ = apply_transforms(input_, size=self._img_size)
        elif type(input_) is str:
            input_ = apply_transforms(load_image(input_), self._img_size)

        if torch.cuda.is_available() and self._use_gpu:
            device = torch.device("cuda")
            self.model = self.model.to(device)
            input_ = input_.to(device)

        # remove previous hooks
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

        # if layer is fc
        if isinstance(layer, nn.modules.Linear):
            num_total_filters = layer.out_features
            self._validate_filter_idx(num_total_filters, filter_idx)

            # if logit is None, default to be index 0.
            if filter_idx is None:
                warnings.warn(UserWarning(
                    f'The target logit is None. Default logit is set to index 0.'))
                filter_idx = 0

            # hook target layer
            self.handlers.append(self._logit_register_forward_hooks(layer, filter_idx))

        # if layer is conv
        elif isinstance(layer, nn.modules.conv.Conv2d):
            num_total_filters = layer.out_channels

            # if filter index is None, do layer visualization
            if filter_idx is None:
                self.handlers.append(self._layer_register_forward_hooks(layer))

            # if filter index is valid number or list, do filter visualization
            elif 'int' in str(type(filter_idx)) or type(filter_idx) == list:
                self._validate_filter_idx(num_total_filters, filter_idx)
                self.handlers.append(self._filter_register_forward_hooks(layer, filter_idx))

            # if else, raise error
            else:
                raise TypeError("filter_idx only can be valid int or non-empty list type!")

        self.handlers.append(self._register_backward_hooks())

        self.gradients = torch.zeros(input_.shape)
        return self._ascent(input_, num_iter)

    def _ascent(self, input_, num_iter):
        """
            optimize input_ via gradient ascent
        """
        output = []
        for i in range(num_iter):
            self.model(input_)
            self.activation.backward()
            self.gradients /= (torch.sqrt(torch.mean(torch.mul(self.gradients, self.gradients))) + 1e-5)
            input_ = input_ + self.gradients * self._lr
            output.append(input_)
        return output

    def visualize(self, layer, filter_idxs=None, input_=None, lr=1., num_iter=30,
                  num_subplots=4, figsize=(4, 4), title='Visualization Result',
                  return_output=True, save_path=None):
        self._lr = lr

        if filter_idxs is None:
            self._visualize_filter(layer,
                                   filter_idxs,
                                   input_,
                                   num_iter=num_iter,
                                   figsize=figsize,
                                   title=title,
                                   save_path=save_path)

        elif type(filter_idxs) is int:
            self._visualize_filter(layer,
                                   filter_idxs,
                                   input_,
                                   num_iter=num_iter,
                                   figsize=figsize,
                                   title=title,
                                   save_path=save_path)

        elif type(filter_idxs) == list and len(filter_idxs) != 0:
            num_total_filters = layer.out_channels
            num_subplots = min(num_total_filters, num_subplots)
            filter_idxs = random.sample(filter_idxs, num_subplots)

            self._visualize_filters(layer,
                                    filter_idxs,
                                    input_,
                                    num_iter=num_iter,
                                    num_subplots=len(filter_idxs),
                                    title=title,
                                    save_path=save_path)

        else:
            raise TypeError("Invalid filter_idxs type!")

        if return_output:
            return self.output

    def _visualize_filters(self, layer, filter_idxs, input_, num_iter, num_subplots, title, save_path=None):

        num_cols = 4
        num_rows = int(np.ceil(num_subplots / num_cols))

        fig = plt.figure(figsize=(16, num_rows * 5))
        plt.title(title)
        plt.axis('off')

        self.output = []

        for i, filter_idx in enumerate(filter_idxs):
            output = self.optimize(layer, filter_idx, input_, num_iter=num_iter)

            self.output.append(output)

            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'filter {filter_idx}')

            ax.imshow(format_for_plotting(standardize_and_clip(output[-1], saturation=0.15, brightness=0.7)))

        plt.subplots_adjust(wspace=0, hspace=0)

        if save_path is not None:
            plt.savefig(save_path)

    def _visualize_filter(self, layer, filter_idx, input_, num_iter, figsize, title, save_path=None):
        self.output = self.optimize(layer, filter_idx, input_, num_iter=num_iter)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(title)

        plt.imshow(format_for_plotting(
            standardize_and_clip(self.output[-1],
                                 saturation=0.15,
                                 brightness=0.7)))

        if save_path is not None:
            plt.savefig(save_path)
