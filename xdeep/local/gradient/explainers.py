"""
Created on Thu Aug 02 16:21:22 2019
@author: Haofan Wang - github.com/haofanwang
"""
from .utils import *

from .backprop.guided_backprop import Backprop
from .backprop.integrate_grad import generate_integrated_grad
from .backprop.smooth_grad import generate_smooth_grad

from .cam.gradcam import GradCAM
from .cam.gradcampp import GradCAMpp
from .cam.scorecam import ScoreCAM


class image_explainer(object):

    def __init__(self, model):
        self.model = model
        self.vanilla_backprop = None
        self.guided_backprop = None
        self.integrate_grad = None
        self.gradcam = None
        self.gradcampp = None

    def explain(self, image, method, viz=True, save_path=None, smooth=None, target_layer_name=None):

        '''
        Args:
            image (Image) - input image
            method (str) - name of visualization method
            viz (bool) - True: visualize, False: not visualize
            smooth (int) - 0:smooth grad, 1:integrate grad
            target_layer_name (str) - name of target layer
            save_path (str) - path to save visualization result
        '''

        norm_image = apply_transforms(image)

        if method == 'backprop':
            explainer = Backprop(self.model, guided=False)
            if smooth == 0:
                self.vanilla_backprop = generate_smooth_grad(explainer, norm_image)
            elif smooth == 1:
                self.vanilla_backprop = generate_integrated_grad(explainer, norm_image)
            else:
                self.vanilla_backprop = explainer.calculate_gradients(norm_image)
            if viz is True:
                visualize(norm_image, self.vanilla_backprop, save_path=save_path, alpha=0.7)

        elif method == 'guided_backprop':
            explainer = Backprop(self.model, guided=True)
            if smooth == 0:
                self.guided_backprop = generate_smooth_grad(explainer, norm_image)
            elif smooth == 1:
                self.guided_backprop = generate_integrated_grad(explainer, norm_image)
            else:
                self.guided_backprop = explainer.calculate_gradients(norm_image)
            if viz is True:
                visualize(norm_image, self.guided_backprop, save_path=save_path, alpha=0.7)

        elif 'cam' in method:
            model_type = str(type(self.model)).split('.')[2]

            if 'vgg' in model_type.lower():
                if target_layer_name is not None:
                    model_dict = dict(type=model_type, arch=self.model, layer_name=target_layer_name,
                                      input_size=(224, 224))
                else:
                    model_dict = dict(type=model_type, arch=self.model, layer_name='features_29',
                                      input_size=(224, 224))

            elif 'resnet' in model_type.lower():
                if target_layer_name is not None:
                    model_dict = dict(type=model_type, arch=self.model, layer_name=target_layer_name,
                                      input_size=(224, 224))
                else:
                    model_dict = dict(type=model_type, arch=self.model, layer_name='layer4',
                                      input_size=(224, 224))

            elif 'densenet' in model_type.lower():
                if target_layer_name is not None:
                    model_dict = dict(type=model_type, arch=self.model, layer_name=target_layer_name,
                                      input_size=(224, 224))
                else:
                    model_dict = dict(type=model_type, arch=self.model,
                                      layer_name='features_denseblock2_denselayer12_norm1',
                                      input_size=(224, 224))

            elif 'alexnet' in model_type.lower():
                if target_layer_name is not None:
                    model_dict = dict(type=model_type, arch=self.model, layer_name=target_layer_name,
                                      input_size=(224, 224))
                else:
                    model_dict = dict(type=model_type, arch=self.model, layer_name='features',
                                      input_size=(224, 224))

            elif 'squeezenet' in model_type.lower():
                if target_layer_name is not None:
                    model_dict = dict(type=model_type, arch=self.model, layer_name=target_layer_name,
                                      input_size=(224, 224))
                else:
                    model_dict = dict(type=model_type, arch=self.model, layer_name='features_12_expand3x3_activation',
                                      input_size=(224, 224))

            else:
                if target_layer_name is None:
                    raise Exception("Not specify target layer")
                else:
                    model_dict = dict(type=model_type, arch=self.model, layer_name=target_layer_name,
                                      input_size=(224, 224))

            if method == 'gradcam':
                gradcam = GradCAM(model_dict)
                mask = gradcam(norm_image)
                if viz is True:
                    visualize(norm_image, mask, save_path=save_path)

            elif method == 'gradcampp':
                gradcampp = GradCAMpp(model_dict)
                mask = gradcampp(norm_image)
                if viz is True:
                    visualize(norm_image, mask)

            elif method == 'scorecam':
                scorecam = ScoreCAM(model_dict)
                mask = scorecam(norm_image)
                if viz is True:
                    visualize(norm_image, mask, save_path=save_path)

        else:
            raise Exception("Invalid method")