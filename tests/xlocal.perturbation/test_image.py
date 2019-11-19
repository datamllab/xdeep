# Please set the folder slim according to your own path.
# Download slim at https://github.com/tensorflow/models/tree/master/research/slim

import os
import tensorflow as tf
from image_util import inception_preprocessing, inception_v3
from skimage.segmentation import slic
import xdeep.xlocal.perturbation.xdeep_image as xdeep_image

def transform_img_fn(path_list):
    out = []
    image_size = inception_v3.default_image_size
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

    names = create_readable_names_for_imagenet_labels()

    processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(processed_images, num_classes=1001, is_training=False)
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

def create_readable_names_for_imagenet_labels():
  """Create a dict mapping label id to human readable string.

  Returns:
      labels_to_names: dictionary where keys are integers from to 1000
      and values are human-readable names.

  We retrieve a synset file, which contains a list of valid synset labels used
  by ILSVRC competition. There is one synset one per line, eg.
          #   n01440764
          #   n01443537
  We also retrieve a synset_to_human_file, which contains a mapping from synsets
  to human-readable names for every synset in Imagenet. These are stored in a
  tsv format, as follows:
          #   n02119247    black fox
          #   n02119359    silver fox
  We assign each synset (in alphabetical order) an integer, starting from 1
  (since 0 is reserved for the background class).

  Code is based on
  https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
  """

  filename = "./image_util/imagenet_lsvrc_2015_synsets.txt"
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  if not num_synsets_in_ilsvrc == 1000:
    raise AssertionError()

  filename = "./image_util/imagenet_metadata.txt"
  synset_to_human_list = open(filename).readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  if not num_synsets_in_all_imagenet == 21842:
    raise AssertionError()

  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    if not len(parts) == 2:
      raise AssertionError()
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  label_index = 1
  labels_to_names = {0: 'background'}
  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  return labels_to_names
