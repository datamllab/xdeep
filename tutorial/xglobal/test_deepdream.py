from xdeep.xglobal.methods.activation import *
import torchvision.models as models


# load model
model = models.vgg16(pretrained=True)

# get layer
layer = model.features[24]

# DeepDream
g_ascent = GradientAscent(model.features)

# image_path = 'jay.jpg', layer = conv5_1, filter_idx = 33,
g_ascent.visualize(layer=layer, filter_idxs=33, input_='images/jay.jpg', title='deepdream',
                   num_iter=50, save_path='results/deepdream.jpg')
