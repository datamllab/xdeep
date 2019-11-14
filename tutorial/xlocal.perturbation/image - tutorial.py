# Please set the folder slim according to your own path.
# Download slim at https://github.com/tensorflow/models/tree/master/research/slim
import sys
sys.path.append("./slim")
import os
import tensorflow as tf
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet
from skimage.segmentation import slic
import xdeep.xlocal.perturbation.xdeep_image as xdeep_image

def transform_img_fn(path_list):
    out = []
    image_size = inception.inception_v3.default_image_size
    for f in path_list:
        with open(f,'rb') as img:
            image_raw = tf.image.decode_jpeg(img.read(), channels=3)
            image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
            out.append(image)
    return session.run([out])[0]

def test_image_data():
    slim = tf.contrib.slim
    tf.reset_default_graph()
    session = tf.Session()

    names = imagenet.create_readable_names_for_imagenet_labels()

    processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    # Please correctly set the model path.
    # Download the model at https://github.com/tensorflow/models/tree/master/research/slim
    checkpoints_dir = 'model'
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
        slim.get_model_variables('InceptionV3'))
    init_fn(session)

    def predict_fn(images):
        return session.run(probabilities, feed_dict={processed_images: images})

    def f(x):
        return x / 2 + 0.5

    class_names = []
    for item in names:
        class_names.append(names[item])

    images = transform_img_fn(['data/violin.JPEG'])
    image = images[0]

    explainer = xdeep_image.ImageExplainer(predict_fn, class_names)

    explainer.explain('lime', image, top_labels=3)
    explainer.show_explanation('lime', deprocess=f, positive_only=False)

    explainer.explain('cle', image, top_labels=3)
    explainer.show_explanation('cle', deprocess=f, positive_only=False)

    explainer.explain('anchor', image)
    explainer.show_explanation('anchor')

    segments_slic = slic(image, n_segments=50, compactness=30, sigma=3)
    explainer.initialize_shap(n_segment=50, segment=segments_slic)
    explainer.explain('shap',image,nsamples=400)
    explainer.show_explanation('shap',deprocess=f)

