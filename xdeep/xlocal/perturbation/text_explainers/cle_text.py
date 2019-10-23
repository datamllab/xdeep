import numpy as np
import matplotlib.pyplot as plt

from ..cle.cle_text import CLETextExplainer

from ..explainer import Explainer


class XDeepCLETextExplainer(Explainer):

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
        """Parameter setter for cle_text.

        # Arguments
            **kwargs: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        class_names = kwargs.pop("class_names", self.class_names)
        self.explainer = CLETextExplainer(class_names=class_names, **kwargs)

    def explain(self, instance, top_labels=None, labels=(1,), care_words=None, spans=(2,), include_original_feature=True, **kwargs):
        """Generate explanation for a prediction using certain method.

        # Arguments
            instance: Str. A raw text string to be explained.
            top_labels: Integer. Number of labels you care about.
            labels: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
            **kwarg: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
        """
        Explainer.explain(self, instance, top_labels=top_labels, labels=labels)
        self.explanation = self.explainer.explain_instance(instance, self.predict_proba, labels=self.labels, top_labels=top_labels,
                                                           care_words=care_words, spans=spans, include_original_feature=include_original_feature, 
                                                           **kwargs)

    def show_explanation(self, span=3, plot=True):
        """Visualization of explanation of cle_text.

        # Arguments
            span: Integer. Each row shows how many features.
            plot: Boolean. Whether plots a figure.
        """
        Explainer.show_explanation(self)
        exp = self.explanation
        labels = self.labels

        print()
        print("CLE Explanation")
        print("Instance: {}".format(self.instance))
        print()
        
        words = exp.domain_mapper.indexed_string.inverse_vocab
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
            for idx in range(len(local_exp)):
                if self.explainer.include_original_feature:
                    if local_exp[idx][0] >= D:
                        index = local_exp[idx][0] - D
                        key = ""
                        for item in parts[index]:
                            key += words[item]+" AND "
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
                print(" {:20} : {:.3f}  |".format(local_exp[idx][0], local_exp[idx][1]), end="")
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