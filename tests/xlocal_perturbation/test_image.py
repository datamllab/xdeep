import torch
import torchvision.models as models
import numpy as np
# from skimage.segmentation import slic
import xdeep.xlocal.perturbation.xdeep_image as xdeep_image
from xdeep.utils import *

image = load_image('tests/xlocal_perturbation/data/violin.JPEG')
image = np.asarray(image)
model = models.vgg16(pretrained=True)


def test_image_data():
  with torch.no_grad():
    names = create_readable_names_for_imagenet_labels()

    class_names = []
    for item in names:
        class_names.append(names[item])

    def predict_fn(x):
      return model(torch.from_numpy(x.reshape(-1, 3, 224, 224)).float()).numpy()

    explainer = xdeep_image.ImageExplainer(predict_fn, class_names)

    explainer.explain('lime', image, top_labels=1, num_samples=100)
    explainer.show_explanation('lime', positive_only=False)

    explainer.explain('cle', image, top_labels=1, num_samples=100)
    explainer.show_explanation('cle', positive_only=False)

    # explainer.explain('anchor', image, threshold=0.7, coverage_samples=5000)
    # explainer.show_explanation('anchor')

    # segments_slic = slic(image, n_segments=10, compactness=30, sigma=3)
    # explainer.initialize_shap(n_segment=10, segment=segments_slic)
    # explainer.explain('shap',image, nsamples=10)
    # explainer.show_explanation('shap')


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

  filename = "tests/xlocal_perturbation/image_util/imagenet_lsvrc_2015_synsets.txt"
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  if not num_synsets_in_ilsvrc == 1000:
    raise AssertionError()

  filename = "tests/xlocal_perturbation/image_util/imagenet_metadata.txt"
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
