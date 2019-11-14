from scipy.sparse.csr import csr_matrix
from IPython.display import display

import shap

from ..explainer import Explainer


class XDeepShapTextExplainer(Explainer):

    def __init__(self, predict_vectorized, class_names, vectorizer, train):
        """Init function. The data pass to 'shap' cannot be str but vector.As a result, you need to pass in the vectorizer.

        # Arguments
            predict_vectorized: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
            vectorizer: Vectorizer. A vectorizer which has 'transform' function that transforms list of str to vector.
            train: Array. Train data, in this case a list of str.
        """
        Explainer.__init__(self, predict_vectorized, class_names)
        self.vectorizer = vectorizer
        self.train = train
        # Initialize explainer
        self.set_parameters()

    def set_parameters(self, **kwargs):
        """Parameter setter for shap. The data pass to 'shap' cannot be str but vector.As a result, you need to pass in the vectorizer.

        # Arguments
            **kwargs: Shap kernel explainer parameter setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
        """
        predict_vectorized = kwargs.pop('predict_vectorized', self.predict_proba)
        vectorizer = kwargs.pop('vectorizer', self.vectorizer)
        vector = kwargs.pop('train', self.train)
        if not isinstance(vector, csr_matrix):
            vector = vectorizer.transform(vector)
        self.explainer = shap.KernelExplainer(predict_vectorized, vector, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Str. A raw text string to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
        """
        if not isinstance(instance, csr_matrix):
            vector = self.vectorizer.transform([instance])
        else:
            vector = instance
        self.instance = vector
        proba = self.predict_proba(self.instance)
        self.original_pred = proba[0]
        if top_labels is not None:
            self.labels = proba.argsort()[0][-top_labels:]
        else:
            self.labels = labels
        self.explanation = self.explainer.shap_values(vector, **kwargs)

    def show_explanation(self, show_in_note_book=True):
        """Visualization of explanation of shap.

        # Arguments
            show_in_note_book: Booleam. Whether show in jupyter notebook.
        """
        Explainer.show_explanation(self)
        shap_values = self.explanation
        expected_value = self.explainer.expected_value
        class_names = self.class_names
        labels = self.labels
        instance = self.instance
        
        shap.initjs()
        print()
        print("Shap Explanation")
        print()

        if len(instance.shape) == 1 or instance.shape[0] == 1:
            for label in labels:
                print("Shap value for label {}:".format(class_names[label]))
                print(shap_values[label])

            if show_in_note_book:
                for label in labels:
                    if isinstance(instance, csr_matrix):
                        display(shap.force_plot(expected_value[label], shap_values[label], instance.A))
                    else:
                        display(shap.force_plot(expected_value[label], shap_values[label], instance))
        else:
            if show_in_note_book:
                shap.summary_plot(shap_values, instance)
