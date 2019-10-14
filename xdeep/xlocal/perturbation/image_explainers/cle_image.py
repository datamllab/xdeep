import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..cle.cle_image import CLEImageExplainer
from ..explainer import Explainer
from ..exceptions import XDeepError

plt.rcParams.update({'font.size': 12})


class XDeepCLEImageExplainer(Explainer):

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
        """Parameter setter for cle_image.

        # Arguments
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        self.explainer = CLEImageExplainer(**kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), care_segments=None, spans=(2,), include_original_feature=True, **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Array. An image to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba, labels=self.labels, top_labels=top_labels,
                                                           care_segments=care_segments, spans=spans,
                                                           include_original_feature=include_original_feature, **kwargs)

    def show_explanation(self, num_features=5, deprocess=None, positive_only=True):
        """Visualization of explanation of lime_image.

        # Arguments
            num_features: Integer. How many features you care about.
            deprocess: Function. A function to deprocess the image.
            positive_only: Boolean. Whether only show feature with positive weight.
        """
        if not isinstance(num_features, int) or num_features < 1:
            raise XDeepError("Number of features should be positive integer.")
        Explainer.show_explanation(self)
        exp = self.explanation
        labels = self.labels
        
        print()
        print("CLE Explanation")
        print()

        colors = list()
        for l in np.linspace(1, 0, 100):
            colors.append((245 / 255, 39 / 255, 87 / 255, l))
        for l in np.linspace(0, 1, 100):
            colors.append((24 / 255, 196 / 255, 93 / 255, l))
        colormap = LinearSegmentedColormap.from_list("CLE", colors)

        # Plot
        fig, axes = plt.subplots(nrows=len(labels), ncols=num_features+1,
                                 figsize=((num_features+1)*3,len(labels)*3))

        assert hasattr(labels, '__len__')
        for n in range(len(labels)):
            # Inverse
            label = labels[-n-1]
            result = exp.intercept[label]
            local_exp = exp.local_exp[label]
            for item in local_exp:
                result += item[1]
            print("Explanation for label {}:".format(self.class_names[label]))
            print("Local Prediction:     {:.3f}".format(result))
            print("Original Prediction:  {:.3f}".format(self.original_pred[label]))
            print()
            img, mask, fs, ws, ms = exp.get_image_and_mask(
                label, positive_only=positive_only,
                num_features=num_features, hide_rest=False
            )
            if deprocess is not None:
                img = deprocess(img)

            mask = np.zeros_like(mask, dtype=np.float64)
            for index in range(len(ws)):
                mask[ms[index] == 1] += ws[index]

            max_val = max(np.max(ws), np.max(mask), abs(np.min(ws)), abs(np.min(mask)))
            min_val = -max_val

            if len(labels) == 1:
                axs = axes
            else:
                axs = axes[n]
            axs[0].axis('off')
            axs[0].set_title("{}".format(self.class_names[label]))
            axs[0].imshow(img, alpha=0.9)
            axs[0].imshow(mask, cmap=colormap, vmin=min_val, vmax=max_val)

            for idx in range(1, num_features+1):
                m = ms[idx-1] * ws[idx-1]
                title = "seg."
                for item in fs[idx-1]:
                    title += str(item)+"&"
                title = title[:-1]
                axs[idx].set_title(title)
                axs[idx].imshow(img, alpha=0.5)
                axs[idx].imshow(m, cmap=colormap, vmin=min_val, vmax=max_val)
                axs[idx].axis('off')
        plt.show()
