from PIL import Image
from IPython.display import display
from skimage.segmentation import mark_boundaries
from matplotlib.colors import LinearSegmentedColormap

import shap
import scipy
import copy
import matplotlib.pyplot as plt
import numpy as np


# Lime
def show_lime_text_explanation(exp, show_in_note_book=True):
	print('\033[1m'+"Lime Explanation"+'\033[0m')
	print()
	print(exp.as_list())
	if show_in_note_book:
		exp.show_in_notebook(text=True)


def show_lime_image_explanation(exp, labels, positive_only=True, num_features=5, hide_rest=False):
    print('\033[1m'+"Lime Explanation"+'\033[0m')
    num = len(labels)
    fig, axs = plt.subplots(num, 1, figsize=(20, 10))
    for idx in range(num):
        label = labels[idx]
        temp, mask = exp.get_image_and_mask(label, positive_only=positive_only,
                                            num_features=num_features, hide_rest=hide_rest)
        axs[idx].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        axs[idx].set_title("Lime Explanation For Label {}".format(label))
    plt.show()


def show_lime_tabular_explanation(exp, show_in_note_book=True):
	print('\033[1m'+"Lime Explanation"+'\033[0m')
	print()
	print(exp.as_list())
	if show_in_note_book:
		exp.show_in_notebook()


# def show_image_prediction(image, preds, names=None, top=2):
# 	plt.imshow(image / 2 + 0.5)
# 	for x in preds.argsort()[0][-top:]:
# 		if names is not None:
# 			print(x, names[x], preds[0, x])
# 		else:
# 			print(x, preds[0, x])
# 	return preds


# Anchor
def show_anchor_text_explanation(exp, instance, predict_fn, show_in_note_book=True, verbose=False):
	if verbose:
		print()
		print('Examples where anchor applies and model predicts same to instance:')
		print()
		print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
		print()
		print('Examples where anchor applies and model predicts different with instance:')
		print()
		print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))

	label = predict_fn([instance]).argsort()[0][-1]
	print('\033[1m'+"Anchor Explanation"+'\033[0m')
	print()
	print('Anchor: %s' % (' AND '.join(exp.names())))
	print('Prediction: {}'.format(label))
	print('Precision: %.2f' % exp.precision())
	if show_in_note_book:
		exp.show_in_notebook()


def show_anchor_image_explanation(segments, exp, image, explainer, predict_fn):
	temp = copy.deepcopy(image)
	temp_img = copy.deepcopy(temp)
	temp[:] = 0
	for x in exp:
		temp[segments == x[0]] = temp_img[segments==x[0]]
	print('Anchor for prediction ', predict_fn(np.expand_dims(image, 0))[0].argmax(), 'confidence', exp[-1][2])
	ShowImageNoAxis(temp)
	if len(exp[-1][3]) != 0:
	    print('Counter Examples:')
	for e in exp[-1][3]:
		data = e[:-1]
		temp = explainer.dummys[e[-1]].copy()
		for x in data.nonzero()[0]:
			temp[segments == x] = image[segments == x]
		ShowImageNoAxis(temp)
		print('Prediction = ', predict_fn(np.expand_dims(temp, 0))[0].argmax())


def ShowImageNoAxis(image, boundaries=None, save=None):
	fig = plt.figure()
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	if boundaries is not None:
		ax.imshow(mark_boundaries(image / 2 + 0.5, boundaries))
	else:
		ax.imshow(image / 2 + .5)
	if save is not None:
		plt.savefig(save)
	plt.show()

def show_anchor_tabular_explanation(exp, instance, predict_fn, show_in_note_book=True, verbose=False):
	if verbose:
		print()
		print('Examples where anchor applies and model predicts same to instance:')
		print()
		print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
		print()
		print('Examples where anchor applies and model predicts different with instance:')
		print()
		print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))

	label = predict_fn([instance]).argsort()[0][-1]
	print('\033[1m'+"Anchor Explanation"+'\033[0m')
	print()
	print('Anchor: %s' % (' AND '.join(exp.names())))
	print('Prediction: {}'.format(label))
	print('Precision: %.2f' % exp.precision())
	if show_in_note_book:
		exp.show_in_notebook()


# Shap
def show_shap_explanation(expected_value, shap_values, x, labels):
	print('\033[1m'+"Shap Explanation"+'\033[0m')
	print()
	for item in labels:
		if isinstance(x, scipy.sparse.csr.csr_matrix):
			display(shap.force_plot(expected_value[item], shap_values[item], x.A))
		else:
			display(shap.force_plot(expected_value[item], shap_values[item], x))


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


def show_shap_image_explanation(top_labels, shap_values, segments_slic, img):
	img = img / 2 + 0.5
	img = Image.fromarray(np.uint8(img*255))
	colors = []
	for l in np.linspace(1, 0, 100):
		colors.append((245 / 255, 39 / 255, 87 / 255, l))
	for l in np.linspace(0, 1, 100):
		colors.append((24 / 255, 196 / 255, 93 / 255, l))
	colormap = LinearSegmentedColormap.from_list("shap", colors)
	# plot our explanations
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
	inds = top_labels
	axes[0].imshow(img)
	axes[0].axis('off')
	max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
	for i in range(len(top_labels)):
		m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
		axes[i + 1].set_title(i)
		axes[i + 1].imshow(img.convert('LA'), alpha=0.15)
		im = axes[i + 1].imshow(m, cmap=colormap, vmin=-max_val, vmax=max_val)
		axes[i + 1].axis('off')
	cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
	cb.outline.set_visible(False)
	plt.show()
