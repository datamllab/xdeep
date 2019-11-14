# XDeep 
## -- *for interpretable deep learning developers*

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2c0aff755250450c90ba167987aaebe5)](https://www.codacy.com/manual/nacoyang/xdeep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=datamllab/xdeep&amp;utm_campaign=Badge_Grade)

XDeep is an open source Python library for **Interpretable Machine Learning**. It is developed by [DATA Lab](http://faculty.cs.tamu.edu/xiahu/index.html) at Texas A&M University. The goal of XDeep is to provide easily accessible interpretation tools for people who want to figure out how deep models work. XDeep provides a variety of methods to interpret a model both locally and globally.

## Installation

To install the package, please use the pip installation as follows (https://pypi.org/project/x-deep/): 

    pip install x-deep

**Note**: currently, XDeep is only compatible with: **Python 3.6**.

## Example

Here is a short example of using the package.

```python
import torchvision.models as models
from xdeep.xlocal.gradient.explainers import *

# load the input image & target deep model
image = load_image('input.jpg')
model = models.vgg16(pretrained=True)

# build the xdeep explainer
model_explainer = ImageInterpreter(model)

# generate the local interpretation
model_explainer.explain(image, method_name='gradcam', target_layer_name='features_29', viz=True) 
```

For detailed tutorial, please check the docs directory of this repository [here](https://github.com/datamllab/xdeep/tree/master/docs).

## Sample Results

<img src="https://github.com/datamllab/xdeep/tree/master/result_img/ensemble_fig.pdf" width="50%" height="45%">

## Cite this work

Fan Yang, Zijian Zhang, Haofan Wang, Yuening Li, Xia Hu "XDeep: An Interpretation Tool for Deep Neural Networks." arXiv:1911.01005, 2019. ([Download](https://arxiv.org/abs/1911.01005))

Biblatex entry:

    @article{yang2019xdeep,
             title={XDeep: An Interpretation Tool for Deep Neural Networks},
             author={Yang, Fan and Zhang, Zijian and Wang, Haofan and Li, Yuening and Hu, Xia},
             journal={arXiv preprint arXiv:1911.01005},
             year={2019}
            }

## DISCLAIMER

Please note that this is a **pre-release** version of the XDeep which is still undergoing final testing before its official release. The website, its software and all contents found on it are provided on an
“as is” and “as available” basis. XDeep does **not** give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. XDeep will **not** be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user’s own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities. Should you encounter any bugs, glitches, lack of functionality or
other problems on the website, please let us know immediately so we
can rectify these accordingly. Your help in this regard is greatly appreciated. 

## Acknowledgements

The authors gratefully acknowledge the XAI program of the Defense Advanced Research Projects Agency (DARPA) administered through grant N66001-17-2-4031; the Texas A&M College of Engineering, and Texas A&M. 
