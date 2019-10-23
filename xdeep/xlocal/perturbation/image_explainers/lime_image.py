# The implementation of LIME refers the original authors' codes in GitHub https://github.com/limetext/lime. 
# The Copyright of algorithm LIME is reserved for (c) 2016, Marco Tulio Correia Ribeiro.
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from lime.lime_image import LimeImageExplainer

from ..explainer import Explainer

plt.rcParams.update({'font.size': 12})


class XDeepLimeImageExplainer(Explainer):

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
        self.explainer = LimeImageExplainer(**kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Array. An image to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba,
                                                           top_labels=top_labels, labels=self.labels, **kwargs)

    def show_explanation(self, deprocess=None, positive_only=True, num_features=5, hide_rest=False):
        """Visualization of explanation of lime_image.

        # Arguments
            deprocess: Function. A function to deprocess the image.
            positive_only: Boolean. Whether only show feature with positive weight.
            num_features: Integer. Numbers of feature you care about.
            hide_rest: Boolean. Whether to hide rest of the image.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        labels = self.labels
        
        print()
        print("LIME Explanation")
        print()

        fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(len(labels)*3, 3))
        
        assert hasattr(labels, '__len__')
        for idx in range(len(labels)):
            # Inverse
            label = labels[-idx-1]
            result = exp.intercept[label]
            local_exp = exp.local_exp[label]
            for item in local_exp:
                result += item[1]
            print("Explanation for label {}:".format(self.class_names[label]))
            print("Local Prediction:     {:.3f}".format(result))
            print("Original Prediction:  {:.3f}".format(self.original_pred[label]))
            print()
            temp, mask = exp.get_image_and_mask(
                label, positive_only=positive_only,
                num_features=num_features, hide_rest=hide_rest
            )
            if deprocess is not None:
                temp = deprocess(temp)
            if len(labels) == 1:
                axs.imshow(mark_boundaries(temp, mask), alpha=0.9)
                axs.axis('off')
                axs.set_title("{}".format(self.class_names[label]))
            else:
                axs[idx].imshow(mark_boundaries(temp, mask), alpha=0.9)
                axs[idx].axis('off')
                axs[idx].set_title("{}".format(self.class_names[label]))
        plt.show()