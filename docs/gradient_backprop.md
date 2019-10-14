<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/backprop/base.py#L5)</span>
## BaseProp class

```python
xdeep.xlocal.gradient.backprop.base.BaseProp(model)
```


Base class for backpropagation.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xlocal/gradient/backprop/guided_backprop.py#L4)</span>
## Backprop class

```python
xdeep.xlocal.gradient.backprop.guided_backprop.Backprop(model, guided=False)
```

Generates vanilla or guided backprop gradients of a target class output w.r.t. an input image.

__Arguments:__

- __model__: torchvision.models. A pretrained model.
- __guided__: bool. If True, perform guided backpropagation. Defaults to False.

__Return:__

Backprop Class.
    

---
## Backprop methods

### calculate_gradients


```python
calculate_gradients(input_, target_class=None, take_max=False, use_gpu=False)
```


Calculate gradient.

__Arguments__

- __input___: torch.Tensor. Preprocessed image with shape (1, C, H, W).
- __target_class__: int. Index of target class. Default to None and use the prediction result as target class.
- __take_max__: bool. Take the maximum across colour channels. Defaults to False.

use_gpu. bool. Use GPU or not. Defaults to False.

__Return:__

Gradient (torch.Tensor) with shape (C, H, W). If take max is True, with shape (1, H, W).
    
----

### generate_integrated_grad


```python
xdeep.xlocal.gradient.backprop.integrate_grad.generate_integrated_grad(explainer, input_, target_class=None, n=25)
```


Generates integrate gradients of given explainer. You can use this with both vanilla
and guided backprop

__Arguments:__

- __explainer__: class. Backprop method.
- __input___: torch.Tensor. Preprocessed image with shape (N, C, H, W).
- __target_class__: int. Index of target class. Default to None.
- __n__: int. Integrate steps. Default to 10.

__Return:__

Integrated gradient (torch.Tensor) with shape (C, H, W).
    
----

### generate_smooth_grad


```python
xdeep.xlocal.gradient.backprop.smooth_grad.generate_smooth_grad(explainer, input_, target_class=None, n=50, mean=0, sigma_multiplier=4)
```


Generates smooth gradients of given explainer.

__Arguments__

- __explainer__: class. Backprop method.
- __input___: torch.Tensor. Preprocessed image with shape (1, C, H, W).
- __target_class__: int. Index of target class. Defaults to None.
- __n__: int. Amount of noisy images used to smooth gradient. Default to 10.
- __mean__: int. Mean value of normal distribution when generating noise. Default to 0.

sigma_multiplier. int. Sigma multiplier when calculating std of noise. Default to 4.

__Return:__

Smooth gradient (torch.Tensor) with shape (C, H, W).
    