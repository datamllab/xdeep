from xdeep.xlocal.gradient.cam.gradcam import *
from xdeep.utils import *
import torchvision.models as models


# prepare image
image_path = 'images/ILSVRC2012_val_00000073.JPEG'
image_name = image_path.split('/')[1]
image = load_image(image_path)
norm_image = apply_transforms(image)

# load a pretrained model
model = models.vgg16(pretrained=True)

# define a parameter dict
model_dict = dict(arch=model, layer_name='features_29', input_size=(224, 224))

# load GradCAM class
gradcam = GradCAM(model_dict)

# generate saliency map
output = gradcam(norm_image)

# save generated result
visualize(norm_image, output, save_path='results/gradcam_' + image_name)
