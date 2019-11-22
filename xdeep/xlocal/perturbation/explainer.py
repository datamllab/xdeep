"""
Base explainer for xdeep.
"""
from .exceptions import XDeepError


class Explainer(object):
    """A base explainer."""

    def __init__(self, predict_proba, class_names):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
        """
        self.labels = list()
        self.explainer = None
        self.explanation = None
        self.instance = None
        self.original_pred = None
        self.class_names = class_names
        self.predict_proba = predict_proba

    def set_parameters(self, **kwargs):
        """Parameter setter.

        # Arguments
            **kwargs: Other parameters that depends on which method you use. For more detail, please check xdeep documentation.
        """
        pass

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Explain function.

        # Arguments
            instance: One instance to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwargs: Other parameters that depends on which method you use. For more detail, please check xdeep documentation.
        """
        self.instance = instance
        try:
            proba = self.predict_proba([instance])[0]
        except:
            proba = self.predict_proba(instance)[0]
        self.original_pred = proba
        if top_labels is not None:
            self.labels = proba.argsort()[-top_labels:]
        else:
            self.labels = labels

    def show_explanation(self, **kwargs):
        """Visualization of explanation.

        # Arguments
            **kwargs: Other parameters that depends on which method you use. For more detail, please check xdeep documentation.
        """
        if not hasattr(self, 'explanation'):
            raise XDeepError("Please call explain function first.")
