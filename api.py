import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask,request,jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('model.keras')

def preprocess_img(path):
    img = image.load_img(path,target_size=(240,240))
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr,axis=0)

    return img_arr

def prediction(img_arr):
    pred = model.predict(img_arr)[0][0]
    result = 'Fake' if pred >= 0.5 else 'Real'
    
    return result,pred

@app.route('/api/predict',methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error':'Image not Uploaded'}),400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join('static',filename)
    file.save(file_path)

    try:
        img_arr = preprocess_img(file_path)
        result, score = prediction(img_arr)
        os.remove(file_path)

        return jsonify({
            'prediction':result,
            'confidence':float(round(score,3))
        }),200
    
    except Exception as e:
        return jsonify({'error':str(e)}),500
    
if __name__ == '__main__':
    app.run(debug=True)