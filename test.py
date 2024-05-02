import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# from tensorflow.keras.applications import VGG16,  DenseNet201, Xception
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

# img = image.load_img('sample/00e6cdc35433bec043975aa5abc8763c.jpg',target_size=(224,224))
img = image.load_img('sample/0b4f5d976cc04a595cabaf5262bc3dadc.jpg',target_size=(224,224))
# img = image.load_img('sample/0b6e86aa52db2272794a4a53d9d8e904c.jpg',target_size=(224,224))
# img = image.load_img('sample/0bf67756a6768dd1ac879c3d66fa35ab.jpg',target_size=(224,224))
# img = image.load_img('sample/00e6cdc35433bec043975aa5abc8763c.jpg',target_size=(224,224))
# img = image.load_img('sample/00e6cdc35433bec043975aa5abc8763c.jpg',target_size=(224,224))
# img = image.load_img('sample/00e6cdc35433bec043975aa5abc8763c.jpg',target_size=(224,224))
img.show()
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)
#correlation, cosine, euclidean, minkowski,
neighbors = NearestNeighbors(n_neighbors=10,algorithm='brute',metric='cosine')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:10]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)

