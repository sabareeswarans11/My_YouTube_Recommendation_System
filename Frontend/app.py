'''
CIS 492/593 Big Data Final Project
My Youtube Recommendation System
Author: Sabareeswaran Shanmugam
Dataset :
credits:https://github.com/prathimacode-hub/ML-ProjectKart
'''

# Frontend Using Python Flask
from __future__ import print_function
import sys

from flask import Flask,render_template,request,redirect

from sab_youtube_recommendation import kmeans_recommend
app = Flask(__name__)

@app.route("/")
def search_topic():
    return render_template("search_topic.html")

@app.route("/result",methods = ['POST', 'GET'])
def result():
    img_url = []
    url1 = []
    if request.method == 'POST':
        searchQuery= request.form['search']
        a,b,c,d=kmeans_recommend(searchQuery)
        search_img_url ="https://www.youtube.com/embed/"+b
        search_url ="https://www.youtube.com/watch?v="+b
        for video in d:
            urls = "https://www.youtube.com/embed/" + video
            img_url.append(urls)
            url_video = "https://www.youtube.com/watch?v=" + video
            url1.append(url_video)

    return render_template("result.html",result=url1,url=img_url,search_result=search_img_url,search_result_vid=search_url,title_video=a,recommended_title=c)

if __name__ == "__main__":
    app.run()
