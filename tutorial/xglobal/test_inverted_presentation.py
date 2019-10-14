from xdeep.xglobal.methods.inverted_representation import *
import torchvision.models as models


# prepare image
image_path = 'images/jay.jpg'
image = load_image(image_path)
norm_image = apply_transforms(image)

# load a pretrained model
model = models.vgg16(pretrained=True)

# define a parameter dict
model_dict = dict(arch=model, layer_name='features_29', input_size=(224, 224))

IR = InvertedRepresentation(model_dict)

# generate saliency map
IR.visualize(norm_image)