# Welcome to XDeep

XDeep is an open source software library for automated Interpretable Machine Learning. It is developed by [DATA Lab](http://faculty.cs.tamu.edu/xiahu/index.html) at Texas A&M University. The ultimate goal of XDeep is to provide easily accessible interpretation tools to people who want to figure out why the model predicts so. XDeep provides a variety of methods to interpret a model locally and globally.

## Installation

To install the package, please use the pip installation as follows:

    pip install x-deep

**Note**: currently, XDeep is only compatible with: **Python 3.6**.

## Interpretation Methods

* Local
    - Gradient
        1. CAM\-Based
            - Grad-CAM
            - Grad-CAM++
            - Score-CAM
        2. Gradient\-Based
            - Vanilla Backpropagation
            - Guided Backpropagation
            - Smooth Gradient
            - Integrate Gradient
    
    - Perturbation
        1. LIME
        2. Anchors
        3. SHAP
        4. CLE

* Global
    - Maximum Activation(filter, layer, logit, deepdream)
    - Inverted Features

## Example

Here is a short example of using the package.

    from xdeep.xlocal.perturbation.xdeep_tabular import TabularExplainer

    iris = sklearn.datasets.load_iris()
    train, test, labels_train, labels_test = 
    sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train, labels_train)

    explainer = TabularExplainer(rf.predict_proba,
    train, feature_names=iris.feature_names, class_names=iris.target_names)
    explainer.set_lime_parameters(discretize_continuous=True)
    explainer.explain('lime', test[0])
    explainer.show_explanation('lime')

For detailed tutorial, please check [here](./tutorial_local) for local ones and [here](./tutorial_global) for global ones. 
