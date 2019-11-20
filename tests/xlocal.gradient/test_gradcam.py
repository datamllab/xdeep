import torchvision.models as models

from xdeep.utils import *
from xdeep.xlocal.gradient.cam.gradcam import *


def test_gradcam():
  
  image_path = 'tests/xlocal.gradient/images/ILSVRC2012_val_00000073.JPEG'
  image_name = image_path.split('/')[1]
  image = load_image(image_path)
  norm_image = apply_transforms(image)

  model = models.vgg16(pretrained=True)
  model_dict = dict(arch=model, layer_name='features_29', input_size=(224, 224))
  
  gradcam = GradCAM(model_dict)
  output = gradcam(norm_image)
  visualize(norm_image, output, save_path='tests/xlocal.gradient/results/gradcam_' + image_name)
