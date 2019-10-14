<span style="float:right;">[[source]](.)</span>
## ImageExplainer class

```python
xdeep.xlocal.perturbation.xdeep_image.ImageExplainer(predict_proba, class_names)
```

Integrated explainer which explains text classifiers.

---
## ImageExplainer methods

### explain


```python
explain(method, instance, top_labels=2, labels=(1,))
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
initialize_shap(n_segment, segment)
```


Init function.

__Arguments__

- __n_segment__: Integer. Number of segments in the image.
- __segment__: Array. An array with 2 dimensions, segments_slic of the image.
    
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

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/image_explainers/lime_image.py#L11)</span>
## XDeepLimeImageExplainer class

```python
xdeep.xlocal.perturbation.image_explainers.lime_image.XDeepLimeImageExplainer(predict_proba, class_names)
```


---
## XDeepLimeImageExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Array. An image to be explained.
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
show_explanation(deprocess=None, positive_only=True, num_features=5, hide_rest=False)
```


Visualization of explanation of lime_image.

__Arguments__

- __deprocess__: Function. A function to deprocess the image.
- __positive_only__: Boolean. Whether only show feature with positive weight.
- __num_features__: Integer. Numbers of feature you care about.
- __hide_rest__: Boolean. Whether to hide rest of the image.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/image_explainers/cle_image.py#L12)</span>
## XDeepCLEImageExplainer class

```python
xdeep.xlocal.perturbation.image_explainers.cle_image.XDeepCLEImageExplainer(predict_proba, class_names)
```


---
## XDeepCLEImageExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,), care_segments=None, spans=(2,), include_original_feature=True)
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Array. An image to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for cle_image.

__Arguments__

- __**kwarg__: Parameters setter. For more detail, please check https://lime-ml.readthedocs.io/en/latest/index.html.
    
---
### show_explanation


```python
show_explanation(num_features=5, deprocess=None, positive_only=True)
```


Visualization of explanation of lime_image.

__Arguments__

- __num_features__: Integer. How many features you care about.
- __deprocess__: Function. A function to deprocess the image.
- __positive_only__: Boolean. Whether only show feature with positive weight.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/image_explainers/anchor_image.py#L11)</span>
## XDeepAnchorImageExplainer class

```python
xdeep.xlocal.perturbation.image_explainers.anchor_image.XDeepAnchorImageExplainer(predict_proba, class_names)
```


---
## XDeepAnchorImageExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.
Anchor does not use top_labels and labels.

__Arguments__

- __instance__: Array. An image to be explained.
- __**kwargs__: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor.
    
---
### set_parameters


```python
set_parameters()
```


Parameter setter for anchor_text.

__Arguments__

- __**kwargs__: Parameters setter. For more detail, please check https://github.com/marcotcr/anchor
    
---
### show_explanation


```python
show_explanation(deprocess=None)
```


Visualization of explanation of anchor_image.

__Arguments__

- __deprocess__: Function. A function to deprocess the image.
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/perturbation/image_explainers/shap_image.py#L11)</span>
## XDeepShapImageExplainer class

```python
xdeep.xlocal.perturbation.image_explainers.shap_image.XDeepShapImageExplainer(predict_proba, class_names, n_segment, segment)
```


---
## XDeepShapImageExplainer methods

### explain


```python
explain(instance, top_labels=None, labels=(1,))
```


Generate explanation for a prediction using certain method.

__Arguments__

- __instance__: Array. An image to be explained.
- __top_labels__: Integer. Number of labels you care about.
- __labels__: Tuple. Labels you care about, if top_labels is not none, it will be replaced by the predicted top labels.
- __**kwarg__: Parameters setter. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.
    
---
### set_parameters


```python
set_parameters(n_segment, segment)
```


Parameter setter for shap_image. For more detail, please check https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py.

__Arguments__

- __n_segment__: Integer. number of segments in the image.
- __segment__: Array. An array with 2 dimensions, segments_slic of the image.
    
---
### show_explanation


```python
show_explanation(deprocess=None)
```


Visualization of explanation of shap.

__Arguments__

- __deprocess__: Function. A function to deprocess the image.
    