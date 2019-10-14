<span style="float:right;">[[source]](.)</span>
## TabularExplainer class

```python
xdeep.xlocal.perturbation.xdeep_tabular.TabularExplainer(predict_proba, class_names, feature_names, train, categorical_features=None, categorical_names=None)
```

Integrated explainer which explains text classifiers.

---
## TabularExplainer methods

### explain


```python
explain(method, instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __method__: Str. The method you want to use.
- __instance__: Instance to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwargs__: Parameters setter. For more detail, please check 'explain_instance' in corresponding method.
    
---
### get_anchor_encoder


```python
get_anchor_encoder(train_data, train_labels, validation_data, validation_labels, discretizer='quartile')
```


Get encoder for tabular data.

If you want to use 'anchor' to explain tabular classifier. You need to get this encoder to encode your data, and train another model.

__Arguments__

- __train_data__: Array. Train data.
- __train_labels__: Array. Train labels.
- __validation_data__: Array. Validation set.
- __validation_labels__: Array. Validation labels.
- __discretizer__: Str. Discretizer for data. Please choose between 'quartile' and 'decile'.

__Return__

A encoder object which has function 'transform'.
    
---
### get_explanation


```python
get_explanation(method)
```


Explanation getter.

__Arguments__

- __method__: Str. The method you want to use.

__Return__

An Explanation object with the corresponding method.
    
---
### initialize_anchor


```python
initialize_anchor(data)
```


Explicit shap initializer.

__Arguments__

- __data__: Array. Full data including train data, validation_data and test data.
    
---
### set_anchor_predict_proba


```python
set_anchor_predict_proba(predict_proba)
```


Anchor predict function setter. 
Because you will get an encoder and train new model, you will need to update the predict function.

__Arguments__

- __predict_proba__: Function. A new classifier prediction probability function.
    
---
### set_parameters


```python
set_parameters(method)
```


Parameter setter.

__Arguments__

- __method__: Str. The method you want to use.
- __**kwargs__: Other parameters that depends on which method you use. For more detail, please check xdeep documentation.
    
---
### show_explanation


```python
show_explanation(method)
```


Visualization of explanation of the corresponding method.

__Arguments__

- __method__: Str. The method you want to use.
- __**kwargs__: parameters setter. For more detail, please check xdeep documentation.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/tabular_explainers/lime_tabular.py#L8)</span>
## XDeepLimeTabularExplainer class

```python
xdeep.xlocal.perturbation.tabular_explainers.lime_tabular.XDeepLimeTabularExplainer(predict_proba, class_names, feature_names, train, categorical_features=None, categorical_names=None)
```


---
## XDeepLimeTabularExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Array. One row of tabular data to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for lime_tabular.

__Arguments__

- __**kwargs__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### show_explanation


```python
show_explanation(span=3)
```


Visualization of explanation of lime_text.

__Arguments__

- __span__: Integer. Each row shows how many features.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/tabular_explainers/cle_tabular.py#L8)</span>
## XDeepCLETabularExplainer class

```python
xdeep.xlocal.perturbation.tabular_explainers.cle_tabular.XDeepCLETabularExplainer(predict_proba, class_names, feature_names, train, categorical_features=None, categorical_names=None)
```


---
## XDeepCLETabularExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,), care_cols=None, spans=(2,), include_original_feature=True)
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Array. One row of tabular data to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for cle_tabular.

__Arguments__

- __**kwargs__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### show_explanation


```python
show_explanation(span=3, plot=True)
```


Visualization of explanation of cle_text.

__Arguments__

- __span__: Boolean. Each row shows how many features.
- __plot__: Boolean. Whether plot a figure.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/tabular_explainers/anchor_tabular.py#L9)</span>
## XDeepAnchorTabularExplainer class

```python
xdeep.xlocal.perturbation.tabular_explainers.anchor_tabular.XDeepAnchorTabularExplainer(class_names, feature_names, data, categorical_names=None)
```


---
## XDeepAnchorTabularExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.
Anchor does not use top_labels and labels.

__Arguments__

- __instance__: Array. One row of tabular data to be explained.
- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### get_anchor_encoder


```python
get_anchor_encoder(train_data, train_labels, validation_data, validation_labels, discretizer='quartile')
```


Get encoder for tabular data.

If you want to use 'anchor' to explain tabular classifier. You need to get this encoder to encode your data, and train another model.

__Arguments__

- __train_data__: Array. Train data.
- __train_labels__: Array. Train labels.
- __validation_data__: Array. Validation set.
- __validation_labels__: Array. Validation labels.
- __discretizer__: Str. Discretizer for data. Please choose between 'quartile' and 'decile'.

__Return__

A encoder object which has function 'transform'.
    
---
### set_anchor_predict_proba


```python
set_anchor_predict_proba(predict_proba)
```


Predict function setter.

__Arguments__

- __predict_proba__: Function. A classifier prediction probability function.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for anchor_tabular.

__Arguments__

- __**kwargs__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### show_explanation


```python
show_explanation(show_in_note_book=True, verbose=True)
```


Visualization of explanation of anchor_tabular.

__Arguments__

- __show_in_note_book__: Boolean. Whether show in jupyter notebook.
- __verbose__: Boolean. Whether print out examples and counter examples.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/tabular_explainers/shap_tabular.py#L9)</span>
## XDeepShapTabularExplainer class

```python
xdeep.xlocal.perturbation.tabular_explainers.shap_tabular.XDeepShapTabularExplainer(predict_proba, class_names, train)
```


---
## XDeepShapTabularExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Array. One row of tabular data to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for shap.

__Arguments__

- __**kwargs__: Parameters setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
    
---
### show_explanation


```python
show_explanation(show_in_note_book=True)
```


Visualization of explanation of shap.

__Arguments__

- __show_in_note_book__: Boolean. Whether show in jupyter notebook.
    