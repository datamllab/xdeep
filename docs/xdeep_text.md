<span style="float:right;">[[source]](.)</span>
## TextExplainer class

```python
xdeep.xlocal.perturbation.xdeep_text.TextExplainer(predict_proba, class_names)
```

Integrated explainer which explains text classifiers.

---
## TextExplainer methods

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
### initialize_shap


```python
initialize_shap(predict_vectorized, vectorizer, train)
```


Explicit shap initializer.

__Arguments__

- __predict_vectorized__: Function. Classifier prediction probability function which need vector passed in.
- __vectorizer__: Vectorizer. A vectorizer which has 'transform' function that transforms list of str to vector.
- __train__: Array. Train data, in this case a list of str.
    
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

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/text_explainers/lime_text.py#L8)</span>
## XDeepLimeTextExplainer class

```python
xdeep.xlocal.perturbation.text_explainers.lime_text.XDeepLimeTextExplainer(predict_proba, class_names)
```


---
## XDeepLimeTextExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Str. A raw text string to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwargs__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for lime_text.

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

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/text_explainers/cle_text.py#L9)</span>
## XDeepCLETextExplainer class

```python
xdeep.xlocal.perturbation.text_explainers.cle_text.XDeepCLETextExplainer(predict_proba, class_names)
```


---
## XDeepCLETextExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,), care_words=None, spans=(2,), include_original_feature=True)
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Str. A raw text string to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for cle_text.

__Arguments__

- __**kwargs__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### show_explanation


```python
show_explanation(span=3, plot=True)
```


Visualization of explanation of cle_text.

__Arguments__

- __span__: Integer. Each row shows how many features.
- __plot__: Boolean. Whether plots a figure.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/text_explainers/anchor_text.py#L9)</span>
## XDeepAnchorTextExplainer class

```python
xdeep.xlocal.perturbation.text_explainers.anchor_text.XDeepAnchorTextExplainer(predict_proba, class_names)
```


---
## XDeepAnchorTextExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.
Anchor does not use top_labels and labels.

__Arguments__

- __instance__: Str. A raw text string to be explained.
- __**kwargs__: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor.
    
---
### set_parameters


```python
set_parameters(nlp=None)
```


Parameter setter for anchor_text.

__Arguments__

- __nlp__: Object. A spacy model object.
- __**kwargs__: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor
    
---
### show_explanation


```python
show_explanation(show_in_note_book=True, verbose=True)
```


Visualization of explanation of anchor_text.

__Arguments__

- __show_in_note_book__: Boolean. Whether show in jupyter notebook.
- __verbose__: Boolean. Whether print out examples and counter examples.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/text_explainers/shap_text.py#L10)</span>
## XDeepShapTextExplainer class

```python
xdeep.xlocal.perturbation.text_explainers.shap_text.XDeepShapTextExplainer(predict_vectorized, class_names, vectorizer, train)
```


---
## XDeepShapTextExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Str. A raw text string to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for shap. The data pass to 'shap' cannot be str but vector.As a result, you need to pass in the vectorizer.

__Arguments__

- __**kwargs__: Shap kernel explainer parameter setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
    
---
### show_explanation


```python
show_explanation(show_in_note_book=True)
```


Visualization of explanation of shap.

__Arguments__

- __show_in_note_book__: Booleam. Whether show in jupyter notebook.
    