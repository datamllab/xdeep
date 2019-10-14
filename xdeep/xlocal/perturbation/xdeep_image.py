"""
Functions for explaining image classifiers.
"""

from .xdeep_base import Wrapper
from .image_explainers.lime_image import XDeepLimeImageExplainer
from .image_explainers.cle_image import XDeepCLEImageExplainer
from .image_explainers.anchor_image import XDeepAnchorImageExplainer
from .image_explainers.shap_image import XDeepShapImageExplainer


class ImageExplainer(Wrapper):
    """Integrated explainer which explains text classifiers."""

    def __init__(self, predict_proba, class_names):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
        """
        Wrapper.__init__(self, predict_proba, class_names=class_names)
        self.methods = ['lime', 'cle', 'anchor', 'shap']
        self.__initialization()

    def __initialization(self):
        """Initializer. Use default parameters to initialize 'lime' and 'anchor' explainer."""
        print("Initialize default 'lime', 'cle', 'anchor' explainers. " + 
              "Please explicitly initialize 'shap' explainer before use it.")
        self.explainers['lime'] = XDeepLimeImageExplainer(self.predict_proba, self.class_names)
        self.explainers['cle'] = XDeepCLEImageExplainer(self.predict_proba, self.class_names)
        self.explainers['anchor'] = XDeepAnchorImageExplainer(self.predict_proba, self.class_names)

    def initialize_shap(self, n_segment, segment):
        """Init function.

        # Arguments
            n_segment: Integer. Number of segments in the image.
            segment: Array. An array with 2 dimensions, segments_slic of the image.
        """
        self.explainers['shap'] = XDeepShapImageExplainer(self.predict_proba, self.class_names, n_segment, segment)

    def explain(self, method, instance, top_labels=2, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            method: Str. The method you want to use.
            instance: Instance to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwargs: Parameters setter. For more detail, please check 'explain_instance' in corresponding method.
        """
        Wrapper.explain(self, method, instance, top_labels=top_labels, labels=labels, **kwargs)

    def show_explanation(self, method, **kwargs):
        """Visualization of explanation of the corresponding method.

        # Arguments
            method: Str. The method you want to use.
            **kwargs: parameters setter. For more detail, please check xdeep documentation.
        """
        Wrapper.show_explanation(self, method, **kwargs)

    def get_explanation(self, method):
        """Explanation getter.

        # Arguments
            method: Str. The method you want to use.
            
        # Return
            An Explanation object with the corresponding method.
        """
        Wrapper.get_explanation(self, method)
