from IPython.display import display
from scipy.sparse.csr import csr_matrix

import shap

from ..explainer import Explainer


class XDeepShapTabularExplainer(Explainer):

    def __init__(self, predict_proba, class_names, train):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
            train: Array. Train data, in this case a list of str.
        """
        Explainer.__init__(self, predict_proba, class_names)
        self.train = train
        # Initialize explainer
        self.set_parameters()

    def set_parameters(self, **kwargs):
        """Parameter setter for shap.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
        """
        predict_proba = kwargs.pop("predict_proba", self.predict_proba)
        train = kwargs.pop("train", self.train)
        self.explainer = shap.KernelExplainer(predict_proba, train, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Array. One row of tabular data to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.shap_values(instance, **kwargs)

    def show_explanation(self, show_in_note_book=True):
        """Visualization of explanation of shap.

        # Arguments
            show_in_note_book: Boolean. Whether show in jupyter notebook.
        """
        Explainer.show_explanation(self)
        shap_values = self.explanation
        expected_value = self.explainer.expected_value
        class_names = self.class_names
        labels = self.labels
        instance = self.instance

        shap.initjs()
        print()
        print("Shap Explanation")
        print()

        assert hasattr(labels, '__len__')
        if len(instance.shape) == 1 or instance.shape[0] == 1:
            for item in labels:
                print("Shap value for label {}:".format(class_names[item]))
                print(shap_values[item])

            if show_in_note_book:
                for item in labels:
                    if isinstance(instance, csr_matrix):
                        display(shap.force_plot(expected_value[item], shap_values[item], instance.A))
                    else:
                        display(shap.force_plot(expected_value[item], shap_values[item], instance))
        else:
            if show_in_note_book:
                shap.summary_plot(shap_values, instance)
