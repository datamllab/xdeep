# The implementation of LIME refers the original authors' codes in GitHub https://github.com/limetext/lime. 
# The Copyright of algorithm LIME is reserved for (c) 2016, Marco Tulio Correia Ribeiro.

import numpy as np
import scipy as sp

from lime import explanation
from lime.lime_text import LimeTextExplainer, IndexedString, IndexedCharacters, TextDomainMapper
from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations


class CLETextExplainer(LimeTextExplainer):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False
                 ):
        """Init function.

            Args:
                kernel_width: kernel width for the exponential kernel.
                kernel: similarity kernel that takes euclidean distances and kernel
                    width as input and outputs weights in (0,1). If None, defaults to
                    an exponential kernel.
                verbose: if true, print local prediction values from linear model
                class_names: list of class names, ordered according to whatever the
                    classifier is using. If not present, class names will be '0',
                    '1', ...
                feature_selection: feature selection method. can be
                    'forward_selection', 'lasso_path', 'none' or 'auto'.
                    See function 'explain_instance_with_data' in lime_base.py for
                    details on what each of the options does.
                split_expression: Regex string or callable. If regex string, will be used with re.split.
                    If callable, the function should return a list of tokens.
                bow: if True (bag of words), will perturb input data by removing
                    all occurrences of individual words or characters.
                    Explanations will be in terms of these words. Otherwise, will
                    explain in terms of word-positions, so that a word may be
                    important the first time it appears and unimportant the second.
                    Only set to false if the classifier uses word order in some way
                    (bigrams, etc), or if you set char_level=True.
                mask_string: String used to mask tokens or characters if bow=False
                    if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                    otherwise.
                random_state: an integer or numpy.RandomState that will be used to
                    generate random numbers. If None, the random state will be
                    initialized using the internal numpy seed.
                char_level: an boolean identifying that we treat each character
                    as an independent occurence in the string
        """
        self.class_names = None
        LimeTextExplainer.__init__(self,
                 kernel_width=kernel_width,
                 kernel=kernel,
                 verbose=verbose,
                 class_names=class_names,
                 feature_selection=feature_selection,
                 split_expression=split_expression,
                 bow=bow,
                 mask_string=mask_string,
                 random_state=random_state,
                 char_level=char_level)

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None,
                         care_words=None,
                         spans=(2,),
                         include_original_feature=True
                         ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        self.care_words = care_words
        self.spans = spans
        self.include_original_feature = include_original_feature
        indexed_string = (IndexedCharacters(
            text_instance, bow=self.bow, mask_string=self.mask_string)
                          if self.char_level else
                          IndexedString(text_instance, bow=self.bow,
                                        split_expression=self.split_expression,
                                        mask_string=self.mask_string))
        domain_mapper = TextDomainMapper(indexed_string)
        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __create_combined_features(self, data):
        assert len(data.shape) == 2
        N, D = data.shape

        if self.care_words is None:
            self.care_words = list(range(D))
        care_words = self.care_words
        spans = self.spans
        include_original_feature = self.include_original_feature

        if not isinstance(care_words, (list, tuple)) or not isinstance(spans, (list, tuple)):
            raise Exception("Parameters 'care_words' and 'spans' should be list or tuple.")
        assert np.max(care_words) < D
        assert len(spans) == 0 or np.max(spans) <= len(care_words)

        self.all_combinations = list()
        for span in spans:
            if span == 1:
                continue
            parts = list(combinations(care_words, span))
            self.all_combinations.extend(parts)
            interaction_num = len(parts)
            temp = np.ones((N, interaction_num), dtype=np.int32)
            for n in range(N):
                for idx in range(interaction_num):
                    part = parts[idx]
                    for item in part:
                        if data[n][item] == 0:
                            temp[n][idx] = 0
                            break
            data = np.concatenate((data, temp), axis=1)

        if not include_original_feature:
            data = data[:, D:]

        return data

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_words()
        sample = self.random_state.randint(1, doc_size, num_samples - 1)
        data = np.ones((num_samples, doc_size))
        features_range = range(doc_size)
        inverse_data = [indexed_string.raw_string()]
        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(features_range, size,
                                                replace=False)
            data[i, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))
        labels = classifier_fn(inverse_data)

        data = self.__create_combined_features(data)

        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances
