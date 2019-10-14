"""
Created on Thu Sep 19 18:37:22 2019
@author: Haofan Wang - github.com/haofanwang
"""

from xdeep.xglobal.methods import *
from xdeep.xglobal.methods.activation import *
from xdeep.xglobal.methods.inverted_representation import *


class GlobalImageInterpreter(object):
    """ Class for global image explanations.

        GlobalImageInterpreter provides unified interface to call different visualization methods.
        The user can use it by several lines. It includes '__init__()' and 'explain()'. The
        users can specify the name of visualization method and target layer. If params are
        not specified, default params will be used.
    """

    def __init__(self, model):
        """Init

        # Arguments
            model: torchvision.models. A pretrained model.
            result: torch.Tensor. Save generated saliency map.

        """
        self.model = model
        self.result = None

    def maximize_activation(self, model):
        """
            Return class of activation maximum.

        """

        return GradientAscent(model)

    def inverted_feature(self, model_dict):
        """
            Return class of inverted representation.
        """
        return InvertedRepresentation(model_dict)

    def explain(self, method_name=None, target_layer=None, target_filter=None, input_=None, num_iter=10, save_path=None):

        """Function to call different visualization methods.

        # Arguments
            method_name: str. The name of interpreter method. Currently, global explanation methods support 'filter',
                                'layer', 'logit', 'deepdream', 'inverted'.
            target_layer: torch.nn.Linear or torch.nn.conv.Conv2d. The objective layer.
            target_filter:  int or list. Index of filter or filters.
            input_: str or Tensor. Path of input image or normalized tensor. Default to be None.
            num_iter: int. Iter times. Default to 10.
            save_path: str. Path to save generated explanations. Default to None.
        """

        method_switch = {"filter": self.maximize_activation,
                         "layer": self.maximize_activation,
                         "logit": self.maximize_activation,
                         "deepdream": self.maximize_activation,
                         "inverted": self.inverted_feature}

        if method_name is None:
            raise TypeError("method_name has to be specified!")

        if method_name == "inverted":
            model_dict = dict(arch=self.model, layer_name=target_layer, input_size=(224, 224))
            explainer = method_switch[method_name](model_dict)

            self.result = explainer.visualize(input_=input_, img_size=(224, 224), alpha_reg_alpha=6,
                                              alpha_reg_lambda=1e-7, tv_reg_beta=2, tv_reg_lambda=1e-8,
                                              n=num_iter, save_path=save_path)
        else:
            if target_layer is None:
                raise TypeError("target_layer cannot be None!")
            elif isinstance(target_layer, str):
                target_layer = find_layer(self.model, target_layer)

            explainer = method_switch[method_name](self.model)

            self.result = explainer.visualize(layer=target_layer, filter_idxs=target_filter,
                                              input_=input_, lr=1, num_iter=num_iter, num_subplots=4,
                                              figsize=(4, 4), title='Visualization Result', return_output=True,
                                              save_path=save_path)

