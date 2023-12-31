import cv2 as cv
import numpy as np
import streamlit as st
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import PIL
from PIL import Image
import os
from joblib import dump, load
import torch
import clip


def ImgToVecKN(img):
    model = load('c:/Users/stalk/GitReps/Laboratory-13.-Search-for-similar-images/model.joblib') 
    gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    arr = [d.pt for d in kp]
    prediction = model.predict(arr)
    hist, bin_edges = np.histogram(prediction, bins=1024)
    return hist

def ImgToVecCLIP(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load('ViT-B/32', device)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        t = model2.encode_image(image_input)[0].numpy()
        t = (t - t.min()) / (t.max() - t.min())
        return t


def decode(str):
    str = str.split(' ')
    #str =map(float, str)
    str = np.asarray(list(map(float, str)))
    #str = str / np.sum(str)
    return list(str)


def get_from_csv(csv_path):
    dataframe = pd.read_csv(csv_path)
    dataframe['vect'] = dataframe['vect'].map(decode)
    return dataframe


def load_rbg(path):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


def Find_Similar(img):
    #df = get_from_csv(path)
    neighbors_kmeans = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    neighbors_kmeans.fit(np.vstack(df['vect'].values), df.index.values)
    
    if CLP:
        #img = Image.open(path)
        img = ImgToVecCLIP(img)
    else:
        img = ImgToVecKN(img)

    #img = img / np.sum(img)
    _, indices_kmeans = neighbors_kmeans.kneighbors([img], n_neighbors=10)
    res = []
    for idx in indices_kmeans.flat:
        res.append(load_rbg(df.loc[idx]['path']))
        #plt.imshow(load_rbg(df.loc[idx]['path']))
        #plt.show()
    return res   

CLP = True
path = ''
if CLP:
    path ='c:/Users/stalk/GitReps/Laboratory-13.-Search-for-similar-images/5DS_CLIP.csv'
else:
    path = 'c:/Users/stalk/GitReps/Laboratory-13.-Search-for-similar-images/5DS.csv'
df = get_from_csv(path)


#img = load_rbg('c:/Users/stalk/GitReps/Laboratory-13.-Search-for-similar-images/VOC2012/JPEGImages/2011_007199.jpg')
#plt.imshow(img)
#sim = Find_Similar(img)
#for sm in sim:
#        plt.imshow(sm)


st.title('Searcher')
file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if file is not None:
    if CLP:
        img = PIL.Image.open(file).convert("RGB")
        img = PIL.ImageOps.exif_transpose(img)
    else:
        buf = np.frombuffer(file.getbuffer(), dtype=np.uint8)
        img = cv.imdecode(buf, cv.IMREAD_COLOR)
    #cv.cvtColor(img, cv.COLOR_BGR2RGB, img)

    #img = load_rbg('c:/Users/stalk/GitReps/Laboratory-13.-Search-for-similar-images/VOC2012/JPEGImages/2011_007199.jpg')

    sim = Find_Similar(img)

    if CLP:
        st.image(img)
    else:
        img_show = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(img_show)
    st.title('Similar:')
    for sm in sim:
        st.image(sm)
