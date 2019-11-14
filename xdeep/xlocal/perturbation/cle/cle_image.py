# The implementation of LIME refers the original authors' codes in GitHub https://github.com/limetext/lime. 
# The Copyright of algorithm LIME is reserved for (c) 2016, Marco Tulio Correia Ribeiro.

from lime.lime_image import LimeImageExplainer, ImageExplanation
from lime.wrappers.scikit_image import SegmentationAlgorithm
from itertools import combinations

import sklearn
import copy
import numpy as np


class CLEImageExplanation(ImageExplanation):

    def __init__(self, image, segments, all_combinations, care_segments=None, spans=(2,), include_original_feature=True):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        ImageExplanation.__init__(self, image, segments)
        self.all_combinations = all_combinations
        self.care_segments = care_segments
        self.spans = spans
        self.include_original_feature = include_original_feature

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: TODO

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        num_segments = np.max(segments)+1
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()

        parts = self.all_combinations

        fs = []
        ws = []
        ms = []
        counter = 0
        if positive_only:
            for x in exp:
                if x[1] > 0 and x[1] > min_weight:
                    ws.append(x[1])

                    if self.include_original_feature:
                        if x[0] < num_segments:
                            fs.append((x[0], ))
                        else:
                            fs.append(parts[x[0]-num_segments])
                    else:
                        fs.append(parts[x[0]])
                    counter += 1
                    if counter == num_features:
                        break

            for f in fs:
                temp_mask = np.zeros(segments.shape, segments.dtype)
                for item in f:
                    temp[segments == item] = image[segments == item].copy()
                    mask[segments == item] = 1
                    temp_mask[segments == item] = 1
                ms.append(temp_mask)
        else:
            for x in exp:
                if np.abs(x[1]) > min_weight:
                    ws.append(x[1])

                    if self.include_original_feature:
                        if x[0] < num_segments:
                            fs.append((x[0], ))
                        else:
                            fs.append(parts[x[0]-num_segments])
                    else:
                        fs.append(parts[x[0]])
                    counter += 1
                    if counter == num_features:
                        break

            for f in fs:
                temp_mask = np.zeros(segments.shape, segments.dtype)
                for item in f:
                    temp[segments == item] = image[segments == item].copy()
                    mask[segments == item] = 1
                    temp_mask[segments == item] = 1
                ms.append(temp_mask)
        return temp, mask, fs, ws, ms


class CLEImageExplainer(LimeImageExplainer):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        LimeImageExplainer.__init__(self, kernel_width=kernel_width, kernel=kernel, 
            verbose=verbose, feature_selection=feature_selection, random_state=random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         care_segments=None,
                         spans=(2,),
                         include_original_feature=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        self.care_segments = care_segments
        self.spans = spans
        self.include_original_feature = include_original_feature

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = CLEImageExplanation(image, segments, self.all_combinations, care_segments=self.care_segments, spans=self.spans,
                                        include_original_feature=self.include_original_feature)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __create_combined_features(self, data):
        assert len(data.shape) == 2
        N, D = data.shape

        if self.care_segments is None:
            self.care_segments = list(range(D))
        care_segments = self.care_segments
        spans = self.spans
        include_original_feature = self.include_original_feature

        if not isinstance(care_segments, (list, tuple)) or not isinstance(spans, (list, tuple)):
            raise Exception("Parameters 'care_segments' and 'spans' should be list or tuple.")
        assert np.max(care_segments) < D
        assert len(spans) == 0 or np.max(spans) <= len(care_segments)

        self.all_combinations = list()
        for span in spans:
            if span == 1:
                continue
            parts = list(combinations(care_segments, span))
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


    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        data = self.__create_combined_features(data)
        return data, np.array(labels)
