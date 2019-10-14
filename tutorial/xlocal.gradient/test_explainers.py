from xdeep.xlocal.gradient.explainers import *
import torchvision.models as models


# load image
image = load_image('images/ILSVRC2012_val_00000073.JPEG')

# load model
model = models.vgg16(pretrained=True)

# load embedded image interpreter
model_explainer = ImageInterpreter(model)

# generate saliency map of method specified by 'method_name'
model_explainer.explain(image, method_name='vallina_backprop', viz=True, save_path='results/bp.jpg')

model_explainer.explain(image, method_name='guided_backprop', viz=True, save_path='results/guided.jpg')

model_explainer.explain(image, method_name='smooth_grad', viz=True, save_path='results/smooth_grad.jpg')

model_explainer.explain(image, method_name='integrate_grad', viz=True, save_path='results/integrate.jpg')

model_explainer.explain(image, method_name='smooth_guided_grad', viz=True, save_path='results/smooth_guided_grad.jpg')

model_explainer.explain(image, method_name='integrate_guided_grad', viz=True, save_path='results/integrate_guided.jpg')

model_explainer.explain(image, method_name='gradcam', target_layer_name='features_29', viz=True, save_path='results/gradcam.jpg')

model_explainer.explain(image, method_name='gradcampp', target_layer_name='features_29', viz=True, save_path='results/gradcampp.jpg')

model_explainer.explain(image, method_name='scorecam', target_layer_name='features_29', viz=True, save_path='results/scorecam.jpg')
