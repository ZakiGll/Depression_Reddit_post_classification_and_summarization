import gradio as gr
import numpy as np
from keras.layers import TextVectorization
from keras.models import load_model
import pandas as pd
import os
from transformers import pipeline

df = pd.read_csv('depression_dataset_reddit_cleaned.csv')
X = df['clean_text']

summarizer = pipeline('summarization')
model = load_model(os.path.join('models', 'Depression_text_classification.h5'))

vectorizer = TextVectorization(max_tokens=250000,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)

def is_depressed(text):

    text_vectorized = vectorizer(text)
    res = model.predict(np.expand_dims(text_vectorized, 0))
    if res > 0.5:
        msg = "The predicted class is : \nDepressed "
    else:
        msg = "The predicted class is : \nNot depressed"

    text_summarized =  summarizer(text)
    msg = msg +"\n"+"Summarization :\n" + text_summarized[0]['summary_text']
    return msg

gr.Interface(is_depressed,"text", "text").launch()