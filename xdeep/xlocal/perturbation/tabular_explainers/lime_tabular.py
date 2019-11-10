# The implementation of LIME refers the original authors' codes in GitHub https://github.com/limetext/lime. 
# The Copyright of algorithm LIME is reserved for (c) 2016, Marco Tulio Correia Ribeiro.
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from ..explainer import Explainer


class XDeepLimeTabularExplainer(Explainer):

    def __init__(self, predict_proba, class_names, feature_names, train, 
                 categorical_features=None, categorical_names=None):
        """Init function.

        # Arguments
            predict_proba: Function. A classifier prediction probability function.
            class_names: List. A list of class names, ordered according to whatever the classifier is using.
            feature_names: List. A list of names (strings) corresponding to the columns in the training data.
            train: Array. Train data.
            categorical_features: List. A list of indices (ints) corresponding to the categorical columns. Everything else will be considered continuous. Values in these columns MUST be integers.
            categorical_names: Dict. A dict which maps from int to list of names, where categorical_names[x][y] represents the name of the yth value of column x.
        """
        Explainer.__init__(self, predict_proba, class_names)
        self.train = train
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        # Initialize explainer
        self.set_parameters()

    def set_parameters(self, **kwargs):
        """Parameter setter for lime_tabular.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        train = kwargs.pop("train", self.train)
        class_names = kwargs.pop("class_names", self.class_names)
        feature_names = kwargs.pop("feature_names", self.feature_names)
        categorical_features = kwargs.pop("categorical_features", self.categorical_features)
        categorical_names = kwargs.pop("categorical_names", self.categorical_names)
        self.explainer = LimeTabularExplainer(train, class_names=class_names,
                                              feature_names=feature_names, categorical_features=categorical_features,
                                              categorical_names=categorical_names, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Array. One row of tabular data to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba, 
                                                           top_labels=top_labels, labels=self.labels, **kwargs)

    def show_explanation(self, span=3):
        """Visualization of explanation of lime_text.

        # Arguments
            span: Integer. Each row shows how many features.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        labels = self.labels
        
        print()
        print("LIME Explanation")
        print("Instance: {}".format(self.instance))
        print()
        
        assert hasattr(labels, '__len__')
        for label in labels:
            result = exp.intercept[label]
            local_exp = exp.local_exp[label]
            for item in local_exp:
                result += item[1]
            print("Explanation for label {}:".format(self.class_names[label]))
            print("Local Prediction:     {:.3f}".format(result))
            print("Original Prediction:  {:.3f}".format(self.original_pred[label]))
            print()
            exp_list = exp.as_list(label=label)
            for idx in range(len(exp_list)):
                print("  {:20} : {:.3f}  |".format(exp_list[idx][0], exp_list[idx][1]), end="")
                if idx % span == span - 1:
                    print()
            print()
            exp.as_pyplot_figure(label=label)
        plt.show()
