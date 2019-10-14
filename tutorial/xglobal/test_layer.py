from xdeep.xglobal.methods.activation import GradientAscent
import torchvision.models as models

# load model
model = models.vgg16(pretrained=True)

# layer
layer = model.features[24]

# Layer Visualizer
g_ascent = GradientAscent(model.features)

# generate layer visualization
g_ascent.visualize(layer, num_iter=100, title='layer visualization', save_path='results/layer.jpg')