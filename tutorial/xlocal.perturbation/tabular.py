import numpy as np
from sklearn.ensemble import RandomForestClassifier
from anchor import utils

dataset_folder = 'data/'
dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder)

c = RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(dataset.train, dataset.labels_train)

import xdeep.local.perturbation.xdeep_tabular as xdeep_tabular

explainer = xdeep_tabular.TabularExplainer(c.predict_proba, ['<=50K', '>50K'], dataset.feature_names, dataset.train[0:50],
                                           categorical_features=dataset.categorical_features, categorical_names=dataset.categorical_names)


explainer.set_parameters('lime', discretize_continuous=True)
explainer.explain('lime', dataset.test[0])
explainer.show_explanation('lime')

explainer.set_parameters('cle', discretize_continuous=True)
explainer.explain('cle', dataset.test[0], spans=(2,))
explainer.show_explanation('cle')

explainer.explain('shap', dataset.test[0])
explainer.show_explanation('shap')

explainer.initialize_anchor(dataset.data)
encoder = explainer.get_anchor_encoder(dataset.train[0:50], dataset.labels_train, dataset.validation, dataset.labels_validation)
c_new = RandomForestClassifier(n_estimators=50, n_jobs=5)
c_new.fit(encoder.transform(dataset.train), dataset.labels_train)
explainer.set_anchor_predict_proba(c_new.predict_proba)
explainer.explain('anchor', dataset.test[0])
explainer.show_explanation('anchor')
