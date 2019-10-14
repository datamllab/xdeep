<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/explainers.py#L18)</span>
## ImageInterpreter class

```python
xdeep.xlocal.gradient.explainers.ImageInterpreter(model)
```

Class for image explanation

ImageInterpreter provides unified interface to call different visualization methods.
The user can use it by several lines. It includes '__init__()' and 'explain()'. The
users can specify the name of visualization method and target layer. If params are
not specified, default params will be used.


---
## ImageInterpreter methods

### explain


```python
explain(image, method_name, viz=True, target_layer_name=None, target_class=None, save_path=None)
```


Function to call different local visualization methods.

__Arguments__

- __image__: str or PIL.Image. The input image to ImageInterpreter. User can directly provide the path of input
                        image, or provide PIL.Image foramt image. For example, image can be './test.jpg' or
                        Image.open('./test.jpg').convert('RGB').
- __method_name__: str. The name of interpreter method. Currently support for 'vallina_backprop', 'guided_backprop',
                'smooth_grad', 'smooth_guided_grad', 'integrate_grad', 'integrate_guided_grad', 'gradcam',
                'gradcampp', 'scorecam'.
- __viz__: bool. Visualize or not. Defaults to True.
- __target_layer_name__: str. The layer to hook gradients and activation map. Defaults to the name of the latest
                        activation map. User can also provide their target layer like 'features_29' or 'layer4'
                        with respect to different network architectures.
- __target_class__: int. The index of target class. Default to be the index of predicted class.
- __save_path__: str. Path to save the saliency map. Default to be None.
    
---
### gradcam


```python
gradcam(model_dict)
```

---
### gradcampp


```python
gradcampp(model_dict)
```

---
### guided_backprop


```python
guided_backprop(model)
```

---
### integrate_grad


```python
integrate_grad(model, input_, guided=False)
```

---
### integrate_guided_grad


```python
integrate_guided_grad(model, input_, guided=True)
```

---
### scorecam


```python
scorecam(model_dict)
```

---
### smooth_grad


```python
smooth_grad(model, input_, guided=False)
```

---
### smooth_guided_grad


```python
smooth_guided_grad(model, input_, guided=True)
```

---
### vallina_backprop


```python
vallina_backprop(model)
```
