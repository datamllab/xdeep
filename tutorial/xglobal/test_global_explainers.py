from xdeep.xglobal.global_explainers import *
import torchvision.models as models


# load model
model = models.vgg16(pretrained=True)

# load global image interpreter
model_explainer = GlobalImageInterpreter(model)

# generate global explanation via activation maximum
model_explainer.explain(method_name='filter', target_layer='features_24', target_filter=20, num_iter=10, save_path='results/filter.jpg')

model_explainer.explain(method_name='layer', target_layer='features_24', num_iter=10, save_path='results/layer.jpg')

model_explainer.explain(method_name='logit', target_layer='features_24', target_filter=20, num_iter=10, save_path='results/logit.jpg')

model_explainer.explain(method_name='deepdream', target_layer='features_24', target_filter=20, input_='images/jay.jpg', num_iter=10, save_path='results/deepdream.jpg')

# generate global explanation via inverted representation
model_explainer.explain(method_name='inverted', target_layer='features_24', input_='images/jay.jpg', num_iter=10)