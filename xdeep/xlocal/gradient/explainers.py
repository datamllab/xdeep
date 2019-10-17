"""
Created on Thu Aug 02 16:21:22 2019
@author: Haofan Wang - github.com/haofanwang
"""


from xdeep.utils import *

from xdeep.xlocal.gradient.backprop.guided_backprop import Backprop
from xdeep.xlocal.gradient.backprop.integrate_grad import generate_integrated_grad
from xdeep.xlocal.gradient.backprop.smooth_grad import generate_smooth_grad

from xdeep.xlocal.gradient.cam.gradcam import GradCAM
from xdeep.xlocal.gradient.cam.gradcampp import GradCAMpp
from xdeep.xlocal.gradient.cam.scorecam import ScoreCAM


class ImageInterpreter(object):

    """ Class for image explanation

        ImageInterpreter provides unified interface to call different visualization methods.
        The user can use it by several lines. It includes '__init__()' and 'explain()'. The
        users can specify the name of visualization method and target layer. If params are
        not specified, default params will be used.
    """

    def __init__(self, model):
        """Init
        
        # Arguments
            model: torchvision.models. A pretrained model.
            result: torch.Tensor. Generated saliency map.

        """
        self.model = model
        self.result = None

    def vanilla_backprop(self, model):
        return Backprop(model, guided=False)

    def guided_backprop(self, model):
        return Backprop(model, guided=True)

    def smooth_grad(self, model, input_, guided=False):
        explainer = Backprop(model, guided=guided)
        return generate_smooth_grad(explainer, input_)

    def smooth_guided_grad(self, model, input_, guided=True):
        explainer = Backprop(model, guided=guided)
        return generate_smooth_grad(explainer, input_)

    def integrate_grad(self, model, input_, guided=False):
        explainer = Backprop(model, guided=guided)
        return generate_integrated_grad(explainer, input_)

    def integrate_guided_grad(self, model, input_, guided=True):
        explainer = Backprop(model, guided=guided)
        return generate_integrated_grad(explainer, input_)

    def gradcam(self, model_dict):
        return GradCAM(model_dict)

    def gradcampp(self, model_dict):
        return GradCAMpp(model_dict)

    def scorecam(self, model_dict):
        return ScoreCAM(model_dict)

    def explain(self, image, method_name, viz=True, target_layer_name=None, target_class=None, save_path=None):

        """Function to call different local visualization methods.

        # Arguments
            image: str or PIL.Image. The input image to ImageInterpreter. User can directly provide the path of input
                                    image, or provide PIL.Image foramt image. For example, image can be './test.jpg' or
                                    Image.open('./test.jpg').convert('RGB').
            method_name: str. The name of interpreter method. Currently support for 'vanilla_backprop', 'guided_backprop',
                            'smooth_grad', 'smooth_guided_grad', 'integrate_grad', 'integrate_guided_grad', 'gradcam',
                            'gradcampp', 'scorecam'.
            viz: bool. Visualize or not. Defaults to True.
            target_layer_name: str. The layer to hook gradients and activation map. Defaults to the name of the latest
                                    activation map. User can also provide their target layer like 'features_29' or 'layer4'
                                    with respect to different network architectures.
            target_class: int. The index of target class. Default to be the index of predicted class.
            save_path: str. Path to save the saliency map. Default to be None.
        """

        method_switch = {"vanilla_backprop": self.vanilla_backprop,
                         "guided_backprop": self.guided_backprop,
                         "smooth_grad": self.smooth_grad,
                         "smooth_guided_grad": self.smooth_guided_grad,
                         "integrate_grad": self.integrate_grad,
                         "integrate_guided_grad": self.integrate_guided_grad,
                         "gradcam": self.gradcam,
                         "gradcampp": self.gradcampp,
                         "scorecam": self.scorecam}

        if type(image) is str:
            image = load_image(image)

        norm_image = apply_transforms(image)

        if method_name not in method_switch.keys():
            raise Exception("Non-support method!", method_name)

        if 'backprop' in method_name:
            explainer = method_switch[method_name](self.model)
            self.result = explainer.calculate_gradients(norm_image, target_class=target_class)
        elif 'cam' in method_name:
            model_dict = dict(arch=self.model, layer_name=target_layer_name, input_size=(224, 224))
            explainer = method_switch[method_name](model_dict)
            self.result = explainer(norm_image, class_idx=target_class)
        else:
            self.result = method_switch[method_name](self.model, norm_image)

        if viz is True:
            visualize(norm_image, self.result, save_path=save_path)

