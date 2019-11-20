import torchvision.models as models
from xdeep.xglobal.global_explainers import *
from xdeep.xglobal.methods.inverted_representation import *


def test_explainer():

    model = models.vgg16(pretrained=True)
    model_explainer = GlobalImageInterpreter(model)

    model_explainer.explain(method_name='filter', target_layer='features_24', target_filter=20, num_iter=10,
                            save_path='./xglobal/results/filter.jpg')
    model_explainer.explain(method_name='layer', target_layer='features_24', num_iter=10, save_path='./xglobal/results/layer.jpg')
    model_explainer.explain(method_name='logit', target_layer='features_24', target_filter=20, num_iter=10,
                            save_path='./xglobal/results/logit.jpg')
    model_explainer.explain(method_name='deepdream', target_layer='features_24', target_filter=20,
                            input_='./xglobal/images/jay.jpg', num_iter=10, save_path='./xglobal/results/deepdream.jpg')
    model_explainer.explain(method_name='inverted', target_layer='features_24', input_='./xglobal/images/jay.jpg', num_iter=10)


def test_filter():
    model = models.vgg16(pretrained=True)
    layer = model.features[24]
    filters = [45, 271, 363, 409]
    g_ascent = GradientAscent(model.features)
    g_ascent.visualize(layer, filters, num_iter=30, title='filter visualization', save_path='./xglobal/results/filter.jpg')


def test_layer():
    model = models.vgg16(pretrained=True)
    layer = model.features[24]
    g_ascent = GradientAscent(model.features)
    g_ascent.visualize(layer, num_iter=100, title='layer visualization', save_path='./xglobal/results/layer.jpg')


def test_deepdream():
    model = models.vgg16(pretrained=True)
    layer = model.features[24]
    g_ascent = GradientAscent(model.features)
    g_ascent.visualize(layer=layer, filter_idxs=33, input_='./xglobal/images/jay.jpg', title='deepdream',num_iter=50, save_path='./xglobal/results/deepdream.jpg')


def test_logit():
    model = models.vgg16(pretrained=True)
    layer = model.classifier[-1]
    logit = 17
    g_ascent = GradientAscent(model)
    g_ascent.visualize(layer, logit, num_iter=30, title='logit visualization', save_path='./xglobal/results/logit.jpg')


def test_inverted():
    image_path = './xglobal/images/jay.jpg'
    image = load_image(image_path)
    norm_image = apply_transforms(image)

    model = models.vgg16(pretrained=True)
    model_dict = dict(arch=model, layer_name='features_29', input_size=(224, 224))
    IR = InvertedRepresentation(model_dict)

    IR.visualize(norm_image)
