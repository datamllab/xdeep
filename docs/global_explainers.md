<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/xdeep/xglobal/global_explainers.py#L11)</span>
## GlobalImageInterpreter class

```python
xdeep.xglobal.global_explainers.GlobalImageInterpreter(model)
```

Class for global image explanations.

GlobalImageInterpreter provides unified interface to call different visualization methods.
The user can use it by several lines. It includes '__init__()' and 'explain()'. The
users can specify the name of visualization method and target layer. If params are
not specified, default params will be used.


---
## GlobalImageInterpreter methods

### explain


```python
explain(method_name=None, target_layer=None, target_filter=None, input_=None, num_iter=10, save_path=None)
```


Function to call different visualization methods.

__Arguments__

- __method_name__: str. The name of interpreter method. Currently, global explanation methods support 'filter',
                    'layer', 'logit', 'deepdream', 'inverted'.
- __target_layer__: torch.nn.Linear or torch.nn.conv.Conv2d. The objective layer.
- __target_filter__:  int or list. Index of filter or filters.
- __input___: str or Tensor. Path of input image or normalized tensor. Default to be None.
- __num_iter__: int. Iter times. Default to 10.
- __save_path__: str. Path to save generated explanations. Default to None.
    
---
### inverted_feature


```python
inverted_feature(model)
```

---
### maximize_activation


```python
maximize_activation(model)
```
