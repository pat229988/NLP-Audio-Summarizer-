import streamlit as st
from random import random
from spacy.lang.en.stop_words import STOP_WORDS
#import en_core_web_sm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import re
from string import punctuation
from heapq import nlargest
#import spacy_streamlit
import configparser
import random
import spacy
import gtts
import os
os.system("spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
#nlp= en_core_web_sm.load()
stopwords = list(STOP_WORDS)
punctuation = punctuation + "\n"
model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
def abst_summary(src_text):
    batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest',return_tensors='pt')
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def word_frequency(doc):
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    return word_frequencies

def sentence_score(sentence_tokens, word_frequencies):
    sentence_score = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_score.keys():
                    sentence_score[sent] = word_frequencies[word.text.lower()] 
                else:
                    sentence_score[sent] += word_frequencies[word.text.lower()]
    return sentence_score

def get_summary(text):

    #text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[.*?]", "", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b[0-9]+\b\s*", " ", text)
    doc = nlp(text)
    
    word_frequencies = word_frequency(doc)
    
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max(word_frequencies.values())
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = sentence_score(sentence_tokens, word_frequencies)
    
    select_length = int(len(sentence_tokens)*0.10)
    if select_length < 1:
        select_length =1
    #print(len(sentence_tokens)*0.10)
    summary  = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    summary = [word.text  for word in summary]
    summary = " ".join(summary)
    #print("sums up:",summary)
    return summary

st.set_page_config(
     page_title="Audio summarizer Web App",
     layout="wide",
     initial_sidebar_state="expanded"
)
st.title("Audio Summaries")
col1, col2 = st.columns(2)

with col1:
    text_ = st.text_area(label="Enter Your Text or story", height=350, placeholder="Enter Your Text or story or your article iit can be of any length")

if st.button("Get Summary"):
    ex_summary = get_summary(text_)
    ab_summary = abst_summary(text_)
    print(ab_summary)
    try:
        with col2:
            st.text_area(label="Extractive Text Summarization (Summary length :{}, Actual Text :{})".format(len(ex_summary),len(text_)),
                        value=ex_summary,
                        height=350)
            #st.text(summary)
            if len(ex_summary)>0:
                ex_tts = gtts.gTTS(ex_summary)
                ex_tts.save("ex_summary.wav")
                ex_audio_file = open('ex_summary.wav', 'rb')
                ex_audio_bytes = ex_audio_file.read()
                data = ex_audio_bytes
                st.audio(data, format="audio/wav", start_time=0, sample_rate=None)
            
            st.text_area(label="Abstractive Text Summarization (Summary length :{}, Actual Text :{})".format(len(ab_summary[0]),len(text_)),
                        value=ab_summary[0],
                        height=350)
            #st.text(summary)
            if len(ab_summary[0])>0:
                ab_tts = gtts.gTTS(ab_summary[0])
                ab_tts.save("ab_summary.wav")
                ab_audio_file = open('ab_summary.wav', 'rb')
                ab_audio_bytes = ab_audio_file.read()
                data = ab_audio_bytes
                st.audio(data, format="audio/wav", start_time=0, sample_rate=None)
    

    except NameError:
        pass
