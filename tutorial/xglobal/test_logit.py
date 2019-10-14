from xdeep.xglobal.methods.activation import GradientAscent
import torchvision.models as models


# load model
model = models.vgg16(pretrained=True)

# logit layer
layer = model.classifier[-1]

# class index, if this a list, logit = [17,242,87,998]
logit = 17

# gradient ascent
g_ascent = GradientAscent(model)

# generate visualization reslut
g_ascent.visualize(layer, logit, num_iter=30, title='logit visualization', save_path='results/logit.jpg')