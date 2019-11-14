# The implementation of LIME refers the original authors' codes in GitHub https://github.com/limetext/lime. 
# The Copyright of algorithm LIME is reserved for (c) 2016, Marco Tulio Correia Ribeiro.

import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from ..explainer import Explainer


class XDeepLimeTextExplainer(Explainer):

    def __init__(self, predict_proba, class_names):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
        """
        Explainer.__init__(self, predict_proba, class_names)
        # Initialize explainer
        self.set_parameters()

    def set_parameters(self, **kwargs):
        """Parameter setter for lime_text.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        class_names = kwargs.pop("class_names", self.class_names)
        self.explainer = LimeTextExplainer(class_names=class_names, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Str. A raw text string to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba,
                                                           top_labels=top_labels, labels=self.labels, **kwargs)

    def show_explanation(self, span=3):
        """Visualization of explanation of lime_text.

        # Arguments
            span: Integer. Each row shows how many features.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        labels = self.labels

        print()
        print("LIME Explanation")
        print("Instance: {}".format(self.instance))
        print()

        assert hasattr(labels, '__len__')
        for label in labels:
            result = exp.intercept[label]
            local_exp = exp.local_exp[label]
            for item in local_exp:
                result += item[1]
            print("Explanation for label {}:".format(self.class_names[label]))
            print("Local Prediction:     {:.3f}".format(result))
            print("Original Prediction:  {:.3f}".format(self.original_pred[label]))
            print()
            exp_list = exp.as_list(label=label)
            for idx in range(len(exp_list)):
                print("  {:20} : {:.3f}  |".format(exp_list[idx][0], exp_list[idx][1]), end="")
                if idx % span == span - 1:
                    print()
            print()
            exp.as_pyplot_figure(label=label)
        plt.show()
