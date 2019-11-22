# The implementation of anchor refers the original authors' codes in GitHub https://github.com/marcotcr/anchor. 
# The Copyright of algorithm anchor is reserved for (c) 2018, Marco Tulio Correia Ribeiro.
import spacy
import numpy as np

from anchor.anchor_text import AnchorText

from ..explainer import Explainer


class XDeepAnchorTextExplainer(Explainer):

    def __init__(self, predict_proba, class_names):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
        """
        Explainer.__init__(self, predict_proba, class_names)
        # Initialize explainer
        self.set_parameters()

    def set_parameters(self, nlp=None, **kwargs):
        """Parameter setter for anchor_text.

        # Arguments
            nlp: Object. A spacy model object.
            **kwargs: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor
        """
        if nlp is None:
            print("Use default nlp = spacy.load('en_core_web_sm') to initialize anchor")
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Cannot load en_core_web_sm")
            print("Please run the command 'python -m spacy download en' as admin, then rerun your code.")

        class_names = kwargs.pop("class_names", self.class_names)
        if nlp is not None:
            self.explainer = AnchorText(nlp, class_names=class_names, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.
           Anchor does not use top_labels and labels.

        # Arguments
            instance: Str. A raw text string to be explained.
            **kwargs: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor.
        """
        def predict_label(x):
            return np.argmax(self.predict_proba(x), axis=1)

        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        try:
            self.labels = self.predict_proba([instance]).argsort()[0][-1:]
        except:
            self.labels = self.predict_proba(instance).argsort()[0][-1:]
        if self.explainer is not None:
            self.explanation = self.explainer.explain_instance(instance, predict_label, **kwargs)

    def show_explanation(self, show_in_note_book=True, verbose=True):
        """Visualization of explanation of anchor_text.

        # Arguments
            show_in_note_book: Boolean. Whether show in jupyter notebook.
            verbose: Boolean. Whether print out examples and counter examples.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        instance = self.instance

        print()
        print("Anchor Explanation")
        print("Instance: {}".format(instance))
        print()
        if verbose:
            print()
            print('Examples where anchor applies and model predicts same to instance:')
            print()
            print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
            print()
            print('Examples where anchor applies and model predicts different with instance:')
            print()
            print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))

        label = self.labels[0]
        print('Prediction: {}'.format(label))
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())
        if show_in_note_book:
            exp.show_in_notebook()
