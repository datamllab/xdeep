# Tutorial

## Local \- Gradient

    import torchvision.models as models
    from xdeep.xlocal.gradient.explainers import *

    # load image
    image = load_image('tutorial/xlocal.gradient/images/ILSVRC2012_val_00000073.JPEG')

    # load model
    model = models.vgg16(pretrained=True)

    # load embedded image interpreter
    model_explainer = ImageInterpreter(model)

    # generate saliency map of method specified by 'method_name'
    model_explainer.explain(image, method_name='vallina_backprop', viz=True, save_path='results/bp.jpg')

    model_explainer.explain(image, method_name='guided_backprop', viz=True, save_path='results/guided.jpg')

    model_explainer.explain(image, method_name='smooth_grad', viz=True, save_path='results/smooth_grad.jpg')

    model_explainer.explain(image, method_name='integrate_grad', viz=True, save_path='results/integrate.jpg')

    model_explainer.explain(image, method_name='gradcam', target_layer_name='features_29', viz=True, save_path='results/gradcam.jpg')

    model_explainer.explain(image, method_name='gradcampp', target_layer_name='features_29', viz=True, save_path='results/gradcampp.jpg')

    model_explainer.explain(image, method_name='scorecam', target_layer_name='features_29', viz=True, save_path='results/scorecam.jpg')

## Local \- Perturbation

### Text

    from xdeep.xlocal.perturbation import xdeep_text

    # Load data and model

    model = load_model()
    dataset = load_data()

    # Start explaining

    text = dataset.test[0]

    explainer = xdeep_text.TextExplainer(model.predict_proba, dataset.class_names)

    explainer.explain('lime', text)
    explainer.show_explanation('lime')

    explainer.explain('cle', text)
    explainer.show_explanation('cle')

    explainer.explain('anchor', text)
    explainer.show_explanation('anchor')

    # for SHAP please check the Jupyter Notebook tutorial on GitHub.

### Tabular

    from xdeep.xlocal.perturbation import xdeep_tabular
    
    # Load data and model

    model = load_model()
    dataset = load_data()

    explainer = xdeep_tabular.TabularExplainer(
    	model.predict_proba, dataset.train, dataset.class_names, dataset.feature_names, 
    	categorical_features=dataset.categorical_features, 
    	categorical_names=dataset.categorical_names)

    explainer.set_parameters('lime', discretize_continuous=True)
    explainer.explain('lime', dataset.test[0])
    explainer.show_explanation('lime')

    explainer.set_parameters('cle', discretize_continuous=True)
    explainer.explain('cle', dataset.test[0])
    explainer.show_explanation('cle')

    explainer.explain('shap', dataset.test[0])
    explainer.show_explanation('shap')

    # for Anchor please check the Jupyter Notebook tutorial on GitHub.

### Image

    from xdeep.local.perturbation import xdeep_image

    # Load data and model

    model = load_model()
    dataset = load_data()

    # Start explaining

    explainer = xdeep_image.ImageExplainer(model.predict_proba, dataset.class_names)

    # deprocess function (If the range of image is [-1, 1], then transfroms it to [0, 1])
    def f(x):
        return x / 2 + 0.5

    image = dataset.test[0]

    explainer.explain('lime', image, top_labels=3)

    explainer.show_explanation('lime', deprocess=f, positive_only=False)

    explainer.explain('cle', image, top_labels=3)

    explainer.show_explanation('cle', deprocess=f, positive_only=False)

    explainer.explain('anchor', image)

    explainer.show_explanation('anchor')

    from skimage.segmentation import slic
    segments_slic = slic(image, n_segments=50, compactness=30, sigma=3)
    explainer.initialize_shap(n_segment=50, segment=segments_slic)
    explainer.explain('shap', image, nsamples=200)
    explainer.show_explanation('shap', deprocess=f)