import numpy as np
import matplotlib.pyplot as plt

from ..cle.cle_tabular import CLETabularExplainer
from ..explainer import Explainer


class XDeepCLETabularExplainer(Explainer):

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
        """Parameter setter for cle_tabular.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        train = kwargs.pop("train", self.train)
        class_names = kwargs.pop("class_names", self.class_names)
        feature_names = kwargs.pop("feature_names", self.feature_names)
        categorical_features = kwargs.pop("categorical_features", self.categorical_features)
        categorical_names = kwargs.pop("categorical_names", self.categorical_names)
        self.explainer = CLETabularExplainer(train, class_names=class_names,
                                               feature_names=feature_names, categorical_features=categorical_features,
                                               categorical_names=categorical_names, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), care_cols=None, spans=(2,), include_original_feature=True, **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Array. One row of tabular data to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba, labels=self.labels,
                                                           top_labels=top_labels,
                                                           care_cols=care_cols, spans=spans,
                                                           include_original_feature=include_original_feature, **kwargs)

    def show_explanation(self, span=3, plot=True):
        """Visualization of explanation of cle_text.

        # Arguments
            span: Boolean. Each row shows how many features.
            plot: Boolean. Whether plot a figure.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        labels = self.labels

        print()
        print("CLE Explanation")
        print("Instance: {}".format(self.instance))
        print()
        
        words = exp.domain_mapper.exp_feature_names
        D = len(words)

        # parts is a list which contains all the combination of features.
        parts = self.explainer.all_combinations

        assert hasattr(labels, '__len__')
        for i in range(len(labels)):
            label = labels[i]
            result = exp.intercept[label]
            local_exp = exp.local_exp[label]
            for item in local_exp:
                result += item[1]
            print("Explanation for label {}:".format(self.class_names[label]))
            print("Local Prediction:     {:.3f}".format(result))
            print("Original Prediction:  {:.3f}".format(self.original_pred[label]))
            print()
            local_exp = exp.local_exp[label]
            for idx in range(len(local_exp)):
                if self.explainer.include_original_feature:
                    if local_exp[idx][0] >= D:
                        index = local_exp[idx][0] - D
                        key = ""
                        for item in parts[index]:
                            key += words[item] + " AND "
                        key = key[:-5]
                        local_exp[idx] = (key, local_exp[idx][1])
                    else:
                        local_exp[idx] = (words[local_exp[idx][0]], local_exp[idx][1])
                else:
                    index = local_exp[idx][0]
                    key = ""
                    for item in parts[index]:
                        key += words[item] + " AND "
                    key = key[:-5]
                    local_exp[idx] = (key, local_exp[idx][1])

            for idx in range(len(local_exp)):
                print("  {:30} : {:.3f}  |".format(local_exp[idx][0], local_exp[idx][1]), end="")
                if idx % span == span - 1:
                    print()
            print()
            if len(local_exp) % span != 0:
                print()

            if plot:
                fig, axs = plt.subplots(nrows=1, ncols=len(labels))
                vals = [x[1] for x in local_exp]
                names = [x[0] for x in local_exp]
                vals.reverse()
                names.reverse()
                pos = np.arange(len(vals)) + .5
                colors = ['green' if x > 0 else 'red' for x in vals]
                if len(labels) == 1:
                    ax = axs
                else:
                    ax = axs[i]
                ax.barh(pos, vals, align='center', color=colors)
                ax.set_yticks(pos)
                ax.set_yticklabels(names)
                title = 'Local explanation for class %s' % self.class_names[label]
                ax.set_title(title)
        if plot:
            plt.show()
