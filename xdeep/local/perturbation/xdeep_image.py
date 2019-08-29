"""
Functions for explaining image classifiers.
"""
from . import util
from .exceptions import XDeepError

from anchor.anchor_image import AnchorImage
from lime.lime_image import LimeImageExplainer


class ImageExplainer(object):

    def __init__(self, predict_proba):
        self.explainers = {}
        self.explanations = {'lime': None, 'anchor': None, 'shap': None}
        self.predict_proba = predict_proba
        self.segment = None
        self.n_segment = None
        self.instance = None
        self.f = None
        self.top_labels = None
        self.__initialization()

    # Use default paramaters to initialize explainers
    def __initialization(self):
        print("Initialize default 'lime', 'anchor' explainers")
        print("If you want to use 'shap', please use 'set_shap_paramaters' to initialize shap explainer first")
        self.set_lime_paramaters()
        self.set_anchor_paramaters()

    def set_lime_paramaters(self, **kwargs):
        self.explainers['lime'] = LimeImageExplainer(**kwargs)

    def set_anchor_paramaters(self, **kwargs):
        self.explainers['anchor'] = AnchorImage(**kwargs)

    def set_shap_paramaters(self, n_segment, segment):
        self.n_segment = n_segment
        self.segment = segment

    @staticmethod
    def _mask_image(zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0, 1))
        out = util.np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i, :, :, :] = image
            for j in range(zs.shape[1]):
                if zs[i, j] == 0:
                    out[i][segmentation == j, :] = background
        return out

    def explain(self, instance, method, **kwargs):
        if method not in self.explanations:
            raise XDeepError("Please input correct explain_method:{} or 'all'".format(self.explainers.keys()))

        self.top_labels = self.predict_proba([instance]).argsort()[0][-2:][::-1]
        self.instance = instance

        if method == 'lime':
            self.explanations['lime'] = self.explainers['lime']\
                .explain_instance(instance[:], self.predict_proba, **kwargs)
            return self.explanations['lime']
        if method == 'anchor':
            threshold = kwargs.pop('threshold', 0.95)
            delta = kwargs.pop('delta', 0.1)
            tau = kwargs.pop('tau', 0.15)
            batch_size = kwargs.pop('batch_size', 100)
            self.explanations['anchor'] = self.explainers['anchor']\
                .explain_instance(instance[:], self.predict_proba, threshold=threshold
                                  , delta=delta, tau=tau, batch_size=batch_size, **kwargs)
            return self.explanations['anchor']
        if method == 'shap':
            if self.n_segment is None or  self.segment is None:
                print("If you want to use shap, please use set_shap_paramaters to initialize shap explainer first")
                return None

            def f(z):
                masked = self._mask_image(z, self.segment, self.instance, 255)
                return self.predict_proba(masked)

            self.f = f
            self.explainers['shap'] = util.shap.KernelExplainer(f, util.np.zeros((1, self.n_segment)), **kwargs)
            self.explanations['shap'] = self.explainers['shap'].shap_values(util.np.ones((1, self.n_segment)), **kwargs)
            return self.explanations['shap']

    def get_explanation(self, method):
        if method not in self.explanations:
            raise XDeepError("Please input correct explain_method:{}".format(self.explainers.keys()))
        return self.explanations[method]

    def show_explanation(self, method='all', **kwargs):
        flag = False
        if method == 'lime' or method == 'all':
            if self.explanations['lime'] is not None:
                if 'labels' not in kwargs.keys():
                    print("Use default top 2 labels")
                    print("You can pass in the labels you care about.")
                labels = kwargs.pop('labels', self.top_labels)
                util.show_lime_image_explanation(self.explanations['lime'], labels, **kwargs)
                flag = True
        if method == 'anchor' or method == 'all':
            if self.explanations['anchor'] is not None:
                util.show_anchor_image_explanation(self.explanations['anchor'][0], self.explanations['anchor'][1]
                                                   , self.instance, self.explainers['anchor'], self.predict_proba)
                flag = True
        if method == 'shap' or method == 'all':
            if self.explanations['shap'] is not None:
                util.shap.initjs()
                exp = self.explanations['shap']
                util.show_shap_image_explanation(self.top_labels, exp, self.segment, self.instance)
                flag = True
        if not flag:
            print("You haven't get explanation yet")
