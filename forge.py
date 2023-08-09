from flask import *
from flask import Flask
from flask import render_template
import os
import torch


from video_prediction import prediction

app = Flask(__name__,template_folder="static/")



@app.route('/')
def home():
    return render_template('ht.html' )
@app.route('/upload',methods=['POST'])
def upload():
    if request.method=='POST':
        video = request.files['video']
        if video:
            # filename = secure_filename(video.filename)
            video.save(os.path.join('videos', 'hello'))
            vid = 'hello'   
            video = vid

            return render_template("ht.html", result = str(prediction.check(vid)))
            # return 'Video uploaded successfully'
        return 'No video was uploaded'
        

    

if __name__=='__main__':
    app.run()
    upload()
