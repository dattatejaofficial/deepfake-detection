import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask,request,redirect,url_for,render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('model.keras')

@app.route('/')
def home():
    return render_template('home.html')

def preprocess_img(path):
    img = image.load_img(path,target_size=(240,240))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array,axis=0)

    return img_array

def prediction(img_arr):
    prediction = model.predict(img_arr)[0][0]

    if prediction >= 0.5:
        res = 'Fake'
    else:
        res = 'Real'
    
    return (res,prediction)


@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['image']
    file_path = os.path.join('static',file.filename)
    file.save(file_path)

    img_arr = preprocess_img(file_path)
    result, score = prediction(img_arr)

    return render_template('prediction.html',result=result,score=round(score,3),img_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)