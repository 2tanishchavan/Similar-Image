import base64

import cv2
from flask import Flask,render_template,request, jsonify
import os
# import pybase64
import base64
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

app = Flask(__name__)

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# @app.route("/", methods=['GET', 'POST'])
# def main():
# 	return render_template("index.html")

# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
# 	if request.method == 'POST':
# 		img = request.files['my_image']
#
# 		img_path = "static/" + img.filename
# 		img.save(img_path)
#
# 		# p = predict_label(img_path)
#
# 	return render_template("index.html", prediction = p, img_path = img_path)


@app.route('/media/upload', methods=['POST'])
def upload_media():
    dd = request.get_json()
    img = print(dd["file"])
    # imgdata = base64.b64decode(img)
    # imgdata = img.decode("base64")
    # image_filename = "image.jpg"
    # handler = open(image_filename, "wb+")
    # handler.write(imgdata)
    # handler.close()
    # decodeit = open('hello_level.jpeg', 'wb')
    # decodeit.write(base64.b64decode((img)))
    # decodeit.close()
    # decode base64 string data
    decoded_data = base64.b64decode((img))
    # write the decoded data back to original format in  file
    img_file = open('image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()
    # with open("image.jpg", "wb") as f:
    #     f.write(imgdata)
    return {"asss":"sdsd"}
    # # if save_uploaded_file(uploaded_file):
    #     # display the file
    #     # display_image = Image.open(uploaded_file)
    #     # st.image(display_image)
    #     # feature extract
    #     features = feature_extraction(os.path.join("image.jpg"),model)
    #     #st.text(features)
    #     # recommendention
    #     indices = recommend(features,feature_list)
    #     # show
    #     # col1,col2,col3,col4,col5 = st.beta_columns(5)
    #     imgarr = []
    #     for file in indices[0][1:2]:
    #         imgarr.append(base64.b64encode(cv2.imread(filenames[file])))
    #     return { "status": "ok", "response": imgarr  }
    # else:
    #     # st.header("Some error occured in file upload")


# @app.route('/get_image')
# def get_image():
#     if request.args.get('type') == '1':
#         filename = 'ok.gif'
#     else:
#         filename = 'error.gif'
#     return send_file(filename, mimetype='image/gif')

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)