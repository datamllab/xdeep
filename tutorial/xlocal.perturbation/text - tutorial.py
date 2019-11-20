import os
import numpy as np
import spacy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
import xdeep.xlocal.perturbation.xdeep_text as xdeep_text

# Please download dataset at 
# http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
# Then reset the path.

def load_polarity(path='./data/rt-polaritydata/rt-polaritydata'):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels

def test_text_data():
    data, labels = load_polarity()
    train, test, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
    train, val, train_labels, val_labels = model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_labels = np.array(val_labels)

    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)
    train_vectors = vectorizer.transform(train)
    test_vectors = vectorizer.transform(test)
    val_vectors = vectorizer.transform(val)

    x = LogisticRegression()
    x.fit(train_vectors, train_labels)
    c = make_pipeline(vectorizer, x)

    explainer = xdeep_text.TextExplainer(c.predict_proba, ['negative', 'positive'])

    text = 'This is a good movie .'

    explainer.explain('lime', text)
    explainer.show_explanation('lime')

    explainer.explain('cle', text)
    explainer.show_explanation('cle')

    try:
        nlp = spacy.load('en_core_web_sm')
        explainer.explain('anchor', text)
        explainer.show_explanation('anchor')
    except OSError:
        pass

    explainer.initialize_shap(x.predict_proba, vectorizer, train[0:10])
    explainer.explain('shap', text)
    explainer.show_explanation('shap')
