"""
Functions for explaining image classifiers.
"""
from . import util
from .exceptions import XDeepError

from anchor.anchor_tabular import AnchorTabularExplainer
from lime.lime_tabular import LimeTabularExplainer


class TabularExplainer(object):

    def __init__(self, predict_proba, train, class_names, feature_names,
                 categorical_features=None, categorical_names=None):
        self.explainers = {}
        self.explanations = {'lime': None, 'anchor': None, 'shap': None}
        self.show_explanations = {'lime': util.show_lime_tabular_explanation, 'anchor': util.show_anchor_tabular_explanation
        , 'shap': util.show_shap_explanation}
        self.predict_proba = predict_proba
        self.train = train
        self.class_names = class_names
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.lime_instance = None
        self.anchor_instance = None
        self.shap_instance = None
        self.top_labels = None
        self._initialization()

    def _initialization(self):
        print("Initialize default 'lime', 'shap' explainer")
        print("If you want to use 'anchor', you need to pass more paramater")
        print("Please call 'set_anchor_paramaters' and 'get_anchor_encoder'"
              ", then use this encoder to process your train data.")
        self.set_lime_paramaters()
        self.set_shap_paramaters()

    def set_lime_paramaters(self, **kwargs):
        self.explainers['lime'] = LimeTabularExplainer(self.train, class_names=self.class_names,
                                                       feature_names=self.feature_names,
                                                       categorical_features=self.categorical_features,
                                                       categorical_names=self.categorical_names,
                                                       **kwargs)

    def set_anchor_paramaters(self, data):
        self.explainers['anchor'] = AnchorTabularExplainer(self.class_names, self.feature_names, data=data,
                                                           categorical_names=self.categorical_names)

    def get_anchor_encoder(self, train_labels, validation_data, validation_labels, discretizer='quartile'):
        if self.explainers['anchor'] is None:
            print("Please call 'set_anchor_paramaters' first")
        self.explainers['anchor'].fit(self.train, train_labels, validation_data, validation_labels,
                                      discretizer=discretizer)
        return self.explainers['anchor'].encoder

    def set_shap_paramaters(self, **kwargs):
        self.explainers['shap'] = util.shap.KernelExplainer(self.predict_proba, self.train, **kwargs)

    def explain(self, instance, method, **kwargs):
        if method not in self.explanations:
            raise XDeepError("Please input correct explain_method:{}".format(self.explainers.keys()))

        self.top_labels = self.predict_proba([instance]).argsort()[0][-2:]

        if method == 'lime':
            self.lime_instance = instance
            labels = kwargs.pop('labels', self.top_labels)
            self.explanations['lime'] = self.explainers['lime']\
                .explain_instance(util.copy.deepcopy(instance), self.predict_proba, labels=labels, **kwargs)
            return self.explanations['lime']
        if method == 'anchor':
            self.anchor_instance = instance
            if 'predict_fn' not in kwargs:
                print("Please pass in your new predict function (predict_fn=...)")
                return None
            self.explanations['anchor'] = self.explainers['anchor']\
                .explain_instance(util.copy.deepcopy(instance), kwargs.pop('predict_fn'), **kwargs)
            return self.explanations['anchor']
        if method == 'shap':
            self.shap_instance = instance
            self.explanations['shap'] = self.explainers['shap'].shap_values(util.copy.deepcopy(instance), **kwargs)
            return self.explanations['shap']

    def get_explanation(self, method):
        if method not in self.explanations:
            raise XDeepError("Please input correct explain_method:{}".format(self.explainers.keys()))
        return self.explanations[method]

    def show_explanation(self, method='all', **kwargs):
        flag = False
        if method == 'lime' or method == 'all':
            if self.explanations['lime'] is not None:
                self.show_explanations['lime'](self.explanations['lime'])
                flag = True
        if method == 'anchor' or method == 'all':
            if self.explanations['anchor'] is not None:
                self.show_explanations['anchor'](self.explanations['anchor'], self.anchor_instance, self.predict_proba)
                flag = True
        if method == 'shap' or method == 'all':
            if self.explanations['shap'] is not None:
                util.shap.initjs()
                exp = self.explanations['shap']
                self.show_explanations['shap'](self.explainers['shap'].expected_value, exp, self.shap_instance, self.top_labels)
                flag = True
        if not flag:
            print("You haven't get explanation yet")
