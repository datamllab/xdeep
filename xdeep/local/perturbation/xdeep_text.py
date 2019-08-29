"""
Functions for explaining text classifiers.
"""
from . import util
from .exceptions import XDeepError

from anchor.anchor_text import AnchorText
from lime.lime_text import LimeTextExplainer

import spacy
import os

class TextExplainer(object):

    def __init__(self, predict_proba, class_names):
        self.explainers = {}
        self.explanations = {'lime': None, 'anchor': None, 'shap': None}
        self.show_explanations = {'lime': util.show_lime_text_explanation, 'anchor': util.show_anchor_text_explanation
        , 'shap': util.show_shap_explanation}
        self.predict_proba = predict_proba
        self.class_names = class_names
        self.lime_instance = None
        self.anchor_instance = None
        self.shap_instance = None
        self.vectorizer = None
        self.top_labels = None
        self.__initialization()

    # Use default paramaters to initialize explainers
    def __initialization(self):
        print("Initialize default 'lime', 'anchor' explainers")
        print("Use 'shap' to explain text data is not recommended")
        print("If you do want to use 'shap', please use 'set_shap_paramaters' to initialize shap explainer first")
        self.set_lime_paramaters()
        self.set_anchor_paramaters(None)

    def set_lime_paramaters(self, **kwargs):
        self.explainers['lime'] = LimeTextExplainer(class_names=self.class_names, **kwargs)

    def set_anchor_paramaters(self, nlp, **kwargs):
        if nlp is None:
            print("Use default nlp = spacy.load('en_core_web_sm') to initialize anchor")
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Download spacy model.")
                os.system("python -m spacy download en")
                try:
                    nlp = spacy.load('en_core_web_sm')
                except OSError:
                    print("Please run the command 'python -m spacy download en' as admin.")
                    print("Then rerun your code.")
        if nlp is not None:
            self.explainers['anchor'] = AnchorText(nlp, self.class_names, **kwargs)

    def set_shap_paramaters(self, vectorizer, predict_vectorized, train, **kwargs):
        self.vectorizer = vectorizer
        vector = train
        if not isinstance(train, util.scipy.sparse.csr.csr_matrix):
            vector = vectorizer.transform(train)
        self.explainers['shap'] = util.shap.KernelExplainer(predict_vectorized, vector, **kwargs)

    def explain(self, instance, method, **kwargs):
        if method not in self.explanations:
            raise XDeepError("Please input correct explain_method:{}".format(self.explainers.keys()))

        self.top_labels = self.predict_proba([instance]).argsort()[0][-1:]

        def predict_label(x):
            return util.np.argmax(self.predict_proba(x), axis=1)

        if method == 'lime':
            self.lime_instance = instance
            labels = kwargs.pop('labels', self.top_labels)
            self.explanations['lime'] = self.explainers['lime']\
                .explain_instance(str(util.copy.deepcopy(instance)), self.predict_proba, labels=labels, **kwargs)
            return self.explanations['lime']
        if method == 'anchor':
            self.anchor_instance = instance
            self.explanations['anchor'] = self.explainers['anchor']\
                .explain_instance(str(util.copy.deepcopy(instance)), predict_label, **kwargs)
            return self.explanations['anchor']
        if method == 'shap':
            if self.explainers['shap'] is None:
                print("Use 'shap' to explain text data is not recommended")
                print("If you do want to use shap, please use set_shap_paramaters to initialize shap explainer first")
                return None
            if not isinstance(instance, util.scipy.sparse.csr.csr_matrix):
                if isinstance(instance, str):
                    vector = self.vectorizer.transform([instance])
                else:
                    vector = self.vectorizer.transform(instance)
            else:
                vector = str(util.copy.deepcopy(instance))
            self.shap_instance = vector
            self.explanations['shap'] = self.explainers['shap'].shap_values(vector, **kwargs)
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

