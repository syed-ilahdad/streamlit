import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🧥 Fashion Recommender System 👗</h1>", unsafe_allow_html=True)
st.markdown("Upload an image of a fashion item and get similar recommendations instantly!", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("👗 Fashion Recommender")
st.sidebar.info("Upload a clothing image and find similar fashion styles using deep learning!")
st.sidebar.markdown("### 💡 About the Project")
st.sidebar.markdown("""This AI-powered fashion recommender system uses deep learning and computer vision to suggest similar outfit styles based on the uploaded image.""")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"File save error: {e}")
        return False
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose a fashion image (jpg/png)...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("✅ Image uploaded successfully!")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Extracting features and searching for recommendations..."):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, feature_list)
        st.subheader("🎯 Recommended Similar Items:")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(filenames[indices[0][i]], caption=f"Recommendation {i+1}", use_container_width=True)
    else:
        st.error("❌ Error uploading the file. Please try again.")