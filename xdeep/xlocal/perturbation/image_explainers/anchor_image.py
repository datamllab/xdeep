# The implementation of anchor refers the original authors' codes in GitHub https://github.com/marcotcr/anchor.
# The Copyright of algorithm anchor is reserved for (c) 2018, Marco Tulio Correia Ribeiro.
import copy
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
from anchor.anchor_image import AnchorImage
from ..explainer import Explainer


class XDeepAnchorImageExplainer(Explainer):

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
        """Parameter setter for anchor_text.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor
        """
        self.explainer = AnchorImage(**kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.
        Anchor does not use top_labels and labels.

        # Arguments
            instance: Array. An image to be explained.
            **kwargs: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        try:
            self.labels = self.predict_proba([instance]).argsort()[0][-1:]
        except:
            self.labels = self.predict_proba(instance).argsort()[0][-1:]
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba, **kwargs)

    def __show_image_no_axis(self, image, boundaries=None, save=None):
        """Show image with no axis

        # Arguments
            image: Array. The image you want to show.
            boundaries: Boundaries
            save: Str. Save path.
        """
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        if boundaries is not None:
            ax.imshow(mark_boundaries(image, boundaries))
        else:
            ax.imshow(image)
        if save is not None:
            plt.savefig(save)
        plt.show()

    def show_explanation(self, deprocess=None):
        """Visualization of explanation of anchor_image.

        # Arguments
            deprocess: Function. A function to deprocess the image.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        segments = exp[0]
        exp = exp[1]
        explainer = self.explainer
        predict_fn = self.predict_proba
        image = self.instance

        print()
        print("Anchor Explanation")
        print()
        if deprocess is not None:
            image = deprocess(copy.deepcopy(image))
        temp = copy.deepcopy(image)
        temp_img = copy.deepcopy(temp)
        temp[:] = np.average(temp)
        for x in exp:
            temp[segments == x[0]] = temp_img[segments == x[0]]
        print('Prediction ', predict_fn(np.expand_dims(image, 0))[0].argmax())
        assert hasattr(exp, '__len__')
        if len(exp) == 0:
            print("Fail to find Anchor")
        else:
            print('Confidence ', exp[-1][2])
            self.__show_image_no_axis(temp)
            if len(exp[-1][3]) != 0:
                print('Counter Examples:')
            for e in exp[-1][3]:
                data = e[:-1]
                temp = explainer.dummys[e[-1]].copy()
                for x in data.nonzero()[0]:
                    temp[segments == x] = image[segments == x]
                self.__show_image_no_axis(temp)
                print('Prediction = ', predict_fn(np.expand_dims(temp, 0))[0].argmax())
