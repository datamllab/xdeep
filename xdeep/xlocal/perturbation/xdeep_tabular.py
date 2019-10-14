"""
Functions for explaining tabular classifiers.
"""

from .xdeep_base import Wrapper
from .exceptions import XDeepError
from .tabular_explainers.lime_tabular import XDeepLimeTabularExplainer
from .tabular_explainers.cle_tabular import XDeepCLETabularExplainer
from .tabular_explainers.anchor_tabular import XDeepAnchorTabularExplainer
from .tabular_explainers.shap_tabular import XDeepShapTabularExplainer


class TabularExplainer(Wrapper):
    """Integrated explainer which explains text classifiers."""

    def __init__(self, predict_proba, class_names, feature_names, train,
                 categorical_features=None, categorical_names=None):
        """Init function.

        # Arguments
            predict_proba: Function. Classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
            feature_names: List. A list of names (strings) corresponding to the columns in the training data.
            train: Array. Train data, in this case a list of tabular data.
            categorical_features: List. A list of indices (ints) corresponding to the categorical columns. Everything else will be considered continuous. Values in these columns MUST be integers.
            categorical_names: Dict. A dict which maps from int to list of names, where categorical_names[x][y] represents the name of the yth value of column x.
        """
        Wrapper.__init__(self, predict_proba, class_names=class_names)
        self.methods = ['lime', 'cle', 'anchor', 'shap']
        self.feature_names = feature_names
        self.train = train
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.__initialization()

    def __initialization(self):
        """Initializer. Use default parameters to initialize 'lime' and 'shap' explainer."""
        print("Initialize default 'lime', 'cle', 'shap' explainers. "
              "Please explicitly initialize 'anchor' explainer before use it.")
        self.explainers['lime'] = XDeepLimeTabularExplainer(self.predict_proba, self.class_names, self.feature_names, 
                                                            self.train, categorical_features=self.categorical_features,
                                                            categorical_names=self.categorical_names)
        self.explainers['cle'] = XDeepCLETabularExplainer(self.predict_proba, self.class_names, self.feature_names, 
                                                            self.train, categorical_features=self.categorical_features,
                                                            categorical_names=self.categorical_names)
        self.explainers['shap'] = XDeepShapTabularExplainer(self.predict_proba, self.class_names, self.train)

    def initialize_anchor(self, data):
        """Explicit shap initializer.

        # Arguments
            data: Array. Full data including train data, validation_data and test data.
        """
        print("If you want to use 'anchor' to explain tabular classifier. "
              "You need to get this encoder to encode your data, and train another model.")
        self.explainers['anchor'] = XDeepAnchorTabularExplainer(self.class_names, self.feature_names, data, 
                                                                categorical_names=self.categorical_names)

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
        if 'anchor' not in self.explainers:
            raise XDeepError("Please initialize anchor explainer first.")
        return self.explainers['anchor'].get_anchor_encoder(train_data, train_labels, 
                                                            validation_data, validation_labels, discretizer='quartile')

    def set_anchor_predict_proba(self, predict_proba):
        """Anchor predict function setter. 
           Because you will get an encoder and train new model, you will need to update the predict function.

        # Arguments
            predict_proba: Function. A new classifier prediction probability function.
        """
        self.explainers['anchor'].set_anchor_predict_proba(predict_proba)

    def explain(self, method, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            method: Str. The method you want to use.
            instance: Instance to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwargs: Parameters setter. For more detail, please check 'explain_instance' in corresponding method.
        """
        Wrapper.explain(self, method, instance, top_labels=top_labels, labels=labels, **kwargs)


    def show_explanation(self, method, **kwargs):
        """Visualization of explanation of the corresponding method.

        # Arguments
            method: Str. The method you want to use.
            **kwargs: parameters setter. For more detail, please check xdeep documentation.
        """
        Wrapper.show_explanation(self, method, **kwargs)

    def get_explanation(self, method):
        """Explanation getter.

        # Arguments
            method: Str. The method you want to use.
            
        # Return
            An Explanation object with the corresponding method.
        """
        Wrapper.get_explanation(self, method)
