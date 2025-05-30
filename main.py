from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from csv import writer
import pandas as pd
from flask_material import Material
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling




UPLOAD_FOLDER = 'static/uploads/'



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'

# Enter your database connection details below


# Enter your database connection details below


# Intialize MySQL

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
class_names = ['advance_glaucoma', 'no_glaucoma', 'normal_glaucoma']
img_height = 500
img_width = 500
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



app = Flask(__name__)


# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'



def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/')
def home():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('about.html')
    # User is not loggedin redirect to login page

@app.route('/home')
def home1():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('home.html')
    # User is not loggedin redirect to login page
@app.route('/pythonlogin/upload_image', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	print(file)
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(UPLOAD_FOLDER, filename))
		path = os.path.join(UPLOAD_FOLDER, filename)
		print(path)
		model = Sequential([
		layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(64, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(3)
		])
		model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		model.load_weights("rentin_acrema.h5")
		img = keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
		img_array = keras.preprocessing.image.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0) # Create a batch
		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])
		class_name = class_names[np.argmax(score)]
		percent_confidence = 100 * np.max(score)
		username="unknown"
		prediciton_details = [username,filename,class_name,percent_confidence]

		class_descriptions = {
			"advance_glaucoma": "Advanced glaucoma detected. Immediate ophthalmologic consultation is recommended. Avoid strain, schedule laser or surgical intervention if advised.",
			"no_glaucoma": "The image doesn't show signs of glaucoma. Keep up regular eye check-ups, especially if you're at risk (diabetes, high BP, family history).",
			"normal_glaucoma": "Early or normal-tension glaucoma detected. Start prescribed medication, avoid caffeine and ensure low-stress environments."
		}

		description = class_descriptions.get(class_name, "No recommendation available.")

		flash(f"This image indicates: **{class_name.upper()}** with **{percent_confidence:.2f}%** confidence.")
		flash(f"Medical Suggestion: {description}")
		return render_template('home.html',
                       filename=filename,
                       class1=class_name,
                       accuracy=percent_confidence,
                       description=description)
	else:
		return redirect(request.url)
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

    
if __name__ =='__main__':
	app.run()
