# Collection of preprocessing functions

from nltk.tokenize import word_tokenize
from transformers import CamembertTokenizer
from transformers import BertTokenizer
from tqdm import tqdm

import numpy as np
import pandas as pd
import re
import string
import unicodedata
import tensorflow as tf
import glob
import os

MAX_LEN = 500
IMG_SHAPE = 224
AUTO = tf.data.experimental.AUTOTUNE
model_bert = 'bert-base-multilingual-cased'
model_camembert = 'camembert-base'

tokenizer_bert = BertTokenizer.from_pretrained(model_bert, do_lowercase=False)
tokenizer_cam = CamembertTokenizer.from_pretrained(model_camembert, do_lowercase=False)


def preprocessing_csv(file):
    file = pd.read_csv(file, index_col=0)
    file = file.reset_index()
    file['filename'] = file.apply(
        lambda x: "datas/images/image_test/image_" + str(x['imageid']) + "_product_" + str(x['productid']) + ".jpg",
        axis=1)
    file['text'] = file.apply(lambda x: str(x['designation']) + ' ' + str(x['description']), axis=1)
    file['text'] = file['text'].str.replace('nan', '')
    file = file.drop(['designation', 'description', 'imageid', 'productid'], axis=1)

    return file


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# preprocess sentences
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub('https?://\S+|www\.\S+', '', w)
    w = re.sub('[%s]' % re.escape(string.punctuation), '', w)
    w = re.sub('\n', '', w)
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)

    mots = word_tokenize(w.strip())
    return ' '.join(mots).strip()


# encode transfomers to tokenizer
def encode(sentences, tokenizer, maxlen=500):
    input_ids = []
    attention_masks = []

    # Pour chaque sentences...
    for sent in tqdm(sentences):
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode. / des fois j'écris en Anglais
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=maxlen,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='np',  # Return Numpy.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into arrays.
    input_ids = np.asarray(input_ids, dtype='int32')
    attention_masks = np.asarray(attention_masks, dtype='int32')

    input_ids = np.squeeze(input_ids)
    attention_masks = np.squeeze(attention_masks)

    return input_ids, attention_masks


@tf.function
# Fonction pour preprocessing des images
def preprocessing_test(img):
    # Lecture et décodage des images:
    img = tf.io.read_file(img)
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])
    img = (img / 255)

    return img


def make_test(x1, x2, x3, x4, x5):
    dataset = tf.data.Dataset.from_tensor_slices((x1, x2, x3, x4, x5)) \
        .map(lambda r, s, t, u, w: [(preprocessing_test(r), s, t, u, w)], num_parallel_calls=AUTO) \
        .batch(1) \
        .prefetch(AUTO)

    return dataset


def fusion_features():
    file_names = glob.glob('datas/images/upload_images/*')
    df_image = pd.DataFrame(file_names, columns=['filename'])
    df_text = pd.read_csv('datas/temp_test.csv')
    df_text = df_text.reset_index(drop=True)
    df = pd.concat([df_image, df_text], axis=1)
    df.to_csv('results/generated.csv', encoding="utf-8", header='True')

    return df


def remove_tempfile():
    file_names = glob.glob('datas/images/upload_images/*')
    for f in file_names:
        os.remove(f)
