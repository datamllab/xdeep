import os

import torch
import torch.optim as optim
from torchvision.utils import save_image

from xdeep.utils import (load_image, apply_transforms, denormalize)


class InvertedRepresentation(object):
    def __init__(self, model_dict):

        """Init

        # Arguments:
            layer_name: str. Name of target layer.
            model: torchvision.models. A pretrained model.
            img_size: tuple. Size of input image, default to (224,224).
        """

        layer_name = model_dict['layer_name']
        self.model = model_dict['arch']
        self.model.eval()

        self.img_size = model_dict['input_size']

        self.activations = dict()
        self.gradients = dict()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        target_layer = self.find_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def find_layer(self, arch, target_layer_name):
        """
            Return target layer
        """

        if target_layer_name is None:
            if 'resnet' in str(type(arch)):
                target_layer_name = 'layer4'
            elif 'alexnet' in str(type(arch)) or 'vgg' in str(type(arch)) or 'squeezenet' in str(type(arch)) or 'densenet' in str(type(arch)):
                target_layer_name = 'features'
            else:
                raise Exception('Invalid layer name! Please specify layer name.', target_layer_name)

        hierarchy = target_layer_name.split('_')

        if hierarchy[0] not in arch._modules.keys():
            raise Exception('Invalid layer name!', target_layer_name)

        target_layer = arch._modules[hierarchy[0]]

        if len(hierarchy) >= 2:
            if hierarchy[1] not in target_layer._modules.keys():
                raise Exception('Invalid layer name!', target_layer_name)
            target_layer = target_layer._modules[hierarchy[1]]

        if len(hierarchy) >= 3:
            if hierarchy[2] not in target_layer._modules.keys():
                raise Exception('Invalid layer name!', target_layer_name)
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) >= 4:
            if hierarchy[3] not in target_layer._modules.keys():
                raise Exception('Invalid layer name!', target_layer_name)
            target_layer = target_layer._modules[hierarchy[3]]

        return target_layer

    def alpha_norm(self, input_, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        return ((input_.view(-1))**alpha).sum()

    def total_variation_norm(self, input_, beta):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check = input_[:, :-1, :-1]
        one_bottom = input_[:, 1:, :-1]
        one_right = input_[:, :-1, 1:]
        total_variation = (((to_check - one_bottom)**2 + (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

    def euclidian_loss(self, org_matrix, target_matrix):
        """
            Euclidian loss is the main loss function.
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    def visualize(self, input_, img_size=(224, 224), alpha_reg_alpha=6, alpha_reg_lambda=1e-7,
                  tv_reg_beta=2, tv_reg_lambda=1e-8, n=20, save_path=None):

        if isinstance(input_, str):
            input_ = load_image(input_)
            input_ = apply_transforms(input_)

        if save_path is None:
            if not os.path.exists('results/inverted'):
                os.makedirs('results/inverted')
            save_path = 'results/inverted'

        # generate random image to be optimized
        opt_img = 1e-1 * torch.randn(1, 3, img_size[0], img_size[1])
        opt_img.requires_grad = True

        # define optimizer
        optimizer = optim.SGD([opt_img], lr=1e4, momentum=0.9)

        # forward propagation to activate hook function
        image_output = self.model(input_)
        input_image_layer_output = self.activations['value']

        for i in range(n):
            optimizer.zero_grad()
            opt_img_output = self.model(opt_img)
            output = self.activations['value']

            euc_loss = 1e-1 * self.euclidian_loss(input_image_layer_output.detach(), output)

            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)
            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img, tv_reg_beta)

            loss = euc_loss + reg_alpha + reg_total_variation

            loss.backward()
            optimizer.step()

            # save image
            if i % 1 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy())
                recreated_img = denormalize(opt_img)
                img_path = os.path.join(save_path, 'Inverted_Iteration_' + str(i) + '.jpg')
                save_image(recreated_img, img_path)

            if i % 40 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
