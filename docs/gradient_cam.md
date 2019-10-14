<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/cam/basecam.py#L3)</span>
## BaseCAM class

```python
xdeep.xlocal.gradient.cam.basecam.BaseCAM(model_dict)
```


Base class for Class Activation Mapping.


---
## BaseCAM methods

### find_layer


```python
find_layer(arch, target_layer_name)
```

---
### forward


```python
forward(input_, class_idx=None, retain_graph=False)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/cam/gradcam.py#L5)</span>
## GradCAM class

```python
xdeep.xlocal.gradient.cam.gradcam.GradCAM(model_dict)
```


GradCAM, inherit from BaseCAM


---
## GradCAM methods

### find_layer


```python
find_layer(arch, target_layer_name)
```

---
### forward


```python
forward(input_, class_idx=None, retain_graph=False)
```


Generates GradCAM result.

__Arguments__

- __input___: torch.Tensor. Preprocessed image with shape (1, C, H, W).
- __class_idx__: int. Index of target class. Defaults to be index of predicted class.

__Return__

Result of GradCAM (torch.Tensor) with shape (1, H, W).
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/cam/gradcampp.py#L6)</span>
## GradCAMpp class

```python
xdeep.xlocal.gradient.cam.gradcampp.GradCAMpp(model_dict)
```


GradCAM++, inherit from BaseCAM


---
## GradCAMpp methods

### find_layer


```python
find_layer(arch, target_layer_name)
```

---
### forward


```python
forward(input, class_idx=None, retain_graph=False)
```


Generates GradCAM++ result.

__Arguments__

- __input___: torch.Tensor. Preprocessed image with shape (1, C, H, W).
- __class_idx__: int. Index of target class. Defaults to be index of predicted class.

__Return__

Result of GradCAM++ (torch.Tensor) with shape (1, H, W).
    
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/cam/scorecam.py#L6)</span>
## ScoreCAM class

```python
xdeep.xlocal.gradient.cam.scorecam.ScoreCAM(model_dict)
```


ScoreCAM, inherit from BaseCAM


---
## ScoreCAM methods

### find_layer


```python
find_layer(arch, target_layer_name)
```

---
### forward


```python
forward(input, class_idx=None, retain_graph=False)
```


Generates ScoreCAM result.

__Arguments__

- __input___: torch.Tensor. Preprocessed image with shape (1, C, H, W).
- __class_idx__: int. Index of target class. Defaults to be index of predicted class.

__Return__

Result of GradCAM (torch.Tensor) with shape (1, H, W).
    