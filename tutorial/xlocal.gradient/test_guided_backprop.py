from xdeep.xlocal.gradient.backprop.guided_backprop import *
from xdeep.utils import *
import torchvision.models as models


# prepare image
image_path = 'images/ILSVRC2012_val_00000073.JPEG'
image_name = image_path.split('/')[1]
image = load_image(image_path)
norm_image = apply_transforms(image)

# load a pretrained model
model = models.vgg16(pretrained=True)

# load Guided_BP class
Guided_BP = Backprop(model, guided=True)

# generate saliency map
output = Guided_BP.calculate_gradients(norm_image)

# save generated result
visualize(norm_image, output, save_path='results/GuidedBP_' + image_name)