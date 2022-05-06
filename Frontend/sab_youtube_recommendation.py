'''
My Youtube Recommendation System
Author: Sabareeswaran Shanmugam
Dataset :https://github.com/sabareeswarans11/My_YouTube_Recommendation_System/blob/master/testdata_with_oneHotEncoding.csv
credits:https://github.com/prathimacode-hub/ML-ProjectKart
'''

# This file act as a backend for the Youtube Recommendation system.
# Using Kmeans Algorithm to get Top ten Recommendation for the Xgbooster classified Test result.
import operator
import random
import pandas as pd
from scipy.spatial import distance
import numpy as np
import json

#import the XGbooster Classified Testset with ground truth to Randomized Recommender Kmeans.
data3 =pd.read_csv('/Users/sabareeswarans/Desktop/BD_Lab/BD_final/Custom_dataset/testdata_with_oneHotEncoding.csv')
data3['groundTruth_category']=data3['groundTruth_category'].fillna(' ')
data3['groundTruth_title']=data3['groundTruth_title'].fillna(' ')

# Calculating Cosine Similarity.

def Similarity(Title1,Title2):
    a = data3.iloc[Title1]
    b = data3.iloc[Title2]
    CategoryA = a['groundTruth_category']
    CategoryB = b['groundTruth_category']
    cata = np.array(json.loads(CategoryA), dtype=float)
    catb = np.array(json.loads(CategoryB), dtype=float)
    CategoryDistance = distance.cosine(cata,catb)
    wordsA = a['groundTruth_title']
    wordsB = b['groundTruth_title']
    worda = np.array(json.loads(wordsA), dtype=float)
    wordb = np.array(json.loads(wordsB), dtype=float)
    wordsDistance = distance.cosine(worda,wordb)
    return CategoryDistance+wordsDistance

def kmeans_recommend(name):
    recommended_results = []
    recommended_youtube_links = []
    new_video = data3[data3['category'].str.contains(name)].iloc[random.randint(1,10)]
    print(new_video)
    def getNeighbors(baseVideo, K):
        distances = []
        for index, video in data3.iterrows():
            if video['video_number'] != baseVideo['video_number']:
                dist = Similarity(baseVideo['video_number'], video['video_number'])
                distances.append((video['video_number'], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors
    K = 10 # Top ten Recommendation for the Xgbooster classified Test result.
    neighbors = getNeighbors(new_video, K)
    for neighbor in neighbors:
        recommended_results.append(data3.iloc[neighbor[0]][1])
    for elt in recommended_results:
        recommended_youtube_links.append(data3.loc[data3['title'] == elt, 'links'].to_string(index=False))
    rem='\n'
    recommended_links = [elem.split(rem, 1)[0] for elem in recommended_youtube_links]
    return new_video.title,new_video.links,recommended_results,recommended_links
