from xdeep.xglobal.methods.activation import GradientAscent
import torchvision.models as models


# load model
model = models.vgg16(pretrained=True)

# layer
layer = model.features[24]

# filters
filters = [45, 271, 363, 409]

# Filter Visualizer
g_ascent = GradientAscent(model.features)

# generate layer visualization
g_ascent.visualize(layer, filters, num_iter=30, title='filter visualization', save_path='results/filter.jpg')
