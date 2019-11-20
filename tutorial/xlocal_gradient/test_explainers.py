import torchvision.models as models
from xdeep.xlocal.gradient.explainers import *


def test_explainer():
  
  image = load_image('images/ILSVRC2012_val_00000073.JPEG')
  model = models.vgg16(pretrained=True)
  model_explainer = ImageInterpreter(model)

  # generate explanations
  model_explainer.explain(image, method_name='vanilla_backprop', viz=True, save_path='results/bp.jpg')
  model_explainer.explain(image, method_name='guided_backprop', viz=True, save_path='results/guided.jpg')
  model_explainer.explain(image, method_name='smooth_grad', viz=True, save_path='results/smooth_grad.jpg')
  model_explainer.explain(image, method_name='integrate_grad', viz=True, save_path='results/integrate.jpg')
  model_explainer.explain(image, method_name='smooth_guided_grad', viz=True, save_path='results/smooth_guided_grad.jpg')
  model_explainer.explain(image, method_name='integrate_guided_grad', viz=True, save_path='results/integrate_guided.jpg')
  model_explainer.explain(image, method_name='gradcam', target_layer_name='features_29', viz=True, save_path='results/gradcam.jpg')
  model_explainer.explain(image, method_name='gradcampp', target_layer_name='features_29', viz=True, save_path='results/gradcampp.jpg')
  model_explainer.explain(image, method_name='scorecam', target_layer_name='features_29', viz=True, save_path='results/scorecam.jpg')
