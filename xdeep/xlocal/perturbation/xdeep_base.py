"""
Base wrapper for xdeep.
"""
from .exceptions import XDeepError


class Wrapper(object):
    """A wrapper to wrap all methods."""

    def __init__(self, predict_proba, class_names):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function, which takes a list of d strings and outputs a (d, k) numpy array with prediction probabilities, where k is the number of classes.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
        """
        self.methods = list()
        self.explainers = dict()

        self.predict_proba = predict_proba
        self.class_names = class_names

    def __initialization(self):
        """Initialize default explainers."""
        pass

    def __check_avaliability(self, method, exp_required=False):
        """Check if the method is valid.

        # Arguments
            method: Str. The method you want to use.
            exp_required: Boolean. Whether check the existence of explanation.
        """
        if method not in self.methods:
            raise XDeepError("Please input correct explain_method from {}.".format(self.methods))
        if method not in self.explainers:
            raise XDeepError("Sorry. You need to initialize {}_explainer first.".format(method))
        if exp_required:
            exp = self.explainers[method].explanation
            if exp is None:
                raise XDeepError("Sorry. You haven't got {} explanation yet.".format(method))

    def set_parameters(self, method, **kwargs):
        """Parameter setter.

        # Arguments
            method: Str. The method you want to use.
            **kwargs: Other parameters that depends on which method you use. For more detail, please check xdeep documentation.
        """
        self.__check_avaliability(method)
        self.explainers[method].set_parameters(**kwargs)

    def explain(self, method, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            method: Str. The method you want to use.
            instance: Instance to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwargs: Parameters setter. For more detail, please check 'explain_instance' in corresponding method.
        """
        # Check validation of parameters
        self.__check_avaliability(method)
        self.explainers[method].explain(instance, top_labels=top_labels, labels=labels, **kwargs)

    def get_explanation(self, method):
        """Explanation getter.

        # Arguments
            method: Str. The method you want to use.
            
        # Return
            An Explanation object with the corresponding method.
        """
        # Check validation of parameters
        self.__check_avaliability(method, exp_required=True)
        exp = self.explainers[method].explanation
        return exp

    def show_explanation(self, method, **kwargs):
        """Visualization of explanation of the corresponding method.

        # Arguments
            method: Str. The method you want to use.
            **kwargs: parameters setter. For more detail, please check xdeep documentation.
        """
        # Check validation of parameters
        self.__check_avaliability(method, exp_required=True)
        self.explainers[method].show_explanation(**kwargs)
