# The implementation of anchor refers the original authors' codes in GitHub https://github.com/marcotcr/anchor. 
# The Copyright of algorithm anchor is reserved for (c) 2018, Marco Tulio Correia Ribeiro.import numpy as np
import numpy as np

from anchor.anchor_tabular import AnchorTabularExplainer

from ..explainer import Explainer
from ..exceptions import XDeepError


class XDeepAnchorTabularExplainer(Explainer):

    def __init__(self, class_names, feature_names, data, categorical_names=None):
        """Init function.

        # Arguments
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
            feature_names: List. list of names (strings) corresponding to the columns in the training data.
            data: Array. Full data including train data, validation_data and test data.
            categorical_names: Dict. A dict which maps from int to list of names, where categorical_names[x][y] represents the name of the yth value of column x.
        """
        Explainer.__init__(self, None, class_names)
        self.feature_names = feature_names
        self.data = data
        self.categorical_names = categorical_names
        # Initialize explainer
        self.set_parameters()

    def set_parameters(self, **kwargs):
        """Parameter setter for anchor_tabular.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        class_names = kwargs.pop("class_names", self.class_names)
        feature_names = kwargs.pop("feature_names", self.feature_names)
        data = kwargs.pop("data", self.data)
        categorical_names = kwargs.pop("categorical_names", self.categorical_names)
        self.explainer = AnchorTabularExplainer(class_names, feature_names, data=data, 
                                                categorical_names=categorical_names)

    def get_anchor_encoder(self, train_data, train_labels, validation_data, validation_labels, discretizer='quartile'):
        """Get encoder for tabular data.

        If you want to use 'anchor' to explain tabular classifier. You need to get this encoder to encode your data, and train another model.

        # Arguments
            train_data: Array. Train data.
            train_labels: Array. Train labels.
            validation_data: Array. Validation set.
            validation_labels: Array. Validation labels.
            discretizer: Str. Discretizer for data. Please choose between 'quartile' and 'decile'.
            
        # Return
            A encoder object which has function 'transform'.
        """
        self.explainer.fit(train_data, train_labels, validation_data, validation_labels,
                           discretizer=discretizer)
        return self.explainer.encoder

    def set_anchor_predict_proba(self, predict_proba):
        """Predict function setter.

        # Arguments
            predict_proba: Function. A classifier prediction probability function.
        """
        self.predict_proba = predict_proba

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.
        Anchor does not use top_labels and labels.

        # Arguments
            instance: Array. One row of tabular data to be explained.
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        if self.predict_proba is None:
            raise XDeepError("Please call set_anchor_predict_proba to pass in your new predict function.")

        def predict_label(x):
            return np.argmax(self.predict_proba(x), axis=1)
        self.instance = instance
        try:
            self.labels = self.predict_proba(self.explainer.encoder.transform([instance])).argsort()[0][-1:]
        except:
            self.labels = self.predict_proba(self.explainer.encoder.transform(instance)).argsort()[0][-1:]
        self.explanation = self.explainer.explain_instance(instance, predict_label, **kwargs)

    def show_explanation(self, show_in_note_book=True, verbose=True):
        """Visualization of explanation of anchor_tabular.

        # Arguments
            show_in_note_book: Boolean. Whether show in jupyter notebook.
            verbose: Boolean. Whether print out examples and counter examples.
        """
        Explainer.show_explanation(self)
        exp = self.explanation

        print()
        print("Anchor Explanation")
        print()
        if verbose:
            print()
            print('Examples where anchor applies and model predicts same to instance:')
            print()
            print(exp.examples(only_same_prediction=True))
            print()
            print('Examples where anchor applies and model predicts different with instance:')
            print()
            print(exp.examples(only_different_prediction=True))

        label = self.labels[0]
        print('Prediction: {}'.format(label))
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())
        if show_in_note_book:
            pass
