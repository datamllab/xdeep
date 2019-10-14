import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

import shap

from ..explainer import Explainer


class XDeepShapImageExplainer(Explainer):

    def __init__(self, predict_proba, class_names, n_segment, segment):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
            n_segment: Integer. Number of segments in the image.
            segment: Array. An array with 2 dimensions, segments_slic of the image.
        """
        Explainer.__init__(self, predict_proba, class_names)
        self.n_segment = n_segment
        self.segment = segment

    @staticmethod
    def _mask_image(zs, segmentation, image, background=None):
        """Mask image function for shap."""
        if background is None:
            background = image.mean((0, 1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i, :, :, :] = image
            for j in range(zs.shape[1]):
                if zs[i, j] == 0:
                    out[i][segmentation == j, :] = background
        return out

    def set_parameters(self, n_segment, segment):
        """Parameter setter for shap_image. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.

        # Arguments
            n_segment: Integer. number of segments in the image.
            segment: Array. An array with 2 dimensions, segments_slic of the image.
        """
        self.n_segment = n_segment
        self.segment = segment

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Array. An image to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)

        def f(z):
            masked = self._mask_image(z, self.segment, self.instance, 255)
            return self.predict_proba(masked)

        self.explainer = shap.KernelExplainer(f, np.zeros((1, self.n_segment)), **kwargs)
        self.explanation = self.explainer.shap_values(np.ones((1, self.n_segment)), **kwargs)

    def __fill_segmentation(self, values, segmentation):
        """Fill segmentation

        # Arguments
            values: Array, weights.
            segmentation: Array. An array with 2 dimensions, segments_slic of the image.

        # Return
            out
        """
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out

    def show_explanation(self, deprocess=None):
        """Visualization of explanation of shap.

        # Arguments
            deprocess: Function. A function to deprocess the image.
        """
        Explainer.show_explanation(self)
        shap_values = self.explanation
        labels = self.labels
        segments_slic = self.segment
        img = self.instance
        
        if deprocess is not None:
            img = deprocess(img)
        img = Image.fromarray(np.uint8(img*255))
        colors = []
        for l in np.linspace(1, 0, 100):
            colors.append((245 / 255, 39 / 255, 87 / 255, l))
        for l in np.linspace(0, 1, 100):
            colors.append((24 / 255, 196 / 255, 93 / 255, l))
        colormap = LinearSegmentedColormap.from_list("shap", colors)
        # Plot our explanations
        fig, axes = plt.subplots(nrows=1, ncols=len(labels)+1, figsize=(12, 4))
        inds = labels
        axes[0].imshow(img)
        axes[0].axis('off')
        max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
        assert hasattr(labels, '__len__')
        for i in range(len(labels)):
            m = self.__fill_segmentation(shap_values[inds[i]][0], segments_slic)
            axes[i + 1].set_title(self.class_names[labels[i]])
            axes[i + 1].imshow(img, alpha=0.5)
            axes[i + 1].imshow(m, cmap=colormap, vmin=-max_val, vmax=max_val)
            axes[i + 1].axis('off')
        plt.show()
