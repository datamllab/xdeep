import torchvision.models as models

from xdeep.utils import *
from xdeep.xlocal.gradient.backprop.guided_backprop import *


def test_bp():

  image_path = 'tests/xlocal.gradient/images/ILSVRC2012_val_00000073.JPEG'
  image_name = image_path.split('/')[1]
  image = load_image(image_path)
  norm_image = apply_transforms(image)

  model = models.vgg16(pretrained=True)
  Guided_BP = Backprop(model, guided=True)
  output = Guided_BP.calculate_gradients(norm_image)
  visualize(norm_image, output, save_path='tests/xlocal.gradient/results/GuidedBP_' + image_name)
