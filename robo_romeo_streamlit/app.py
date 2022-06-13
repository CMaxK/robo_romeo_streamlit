import streamlit as st
import numpy as np
from PIL import Image
import os
from model import build_autoencoder #links with model building .py file

st.title("Robo-Romeo❤️")
st.markdown("Upload your image - Robo-Romeo will provide you with a caption and romantic poetry")


@st.cache(allow_output_mutation=True) #cache the model at first loading
def load_model_cache():

    model = build_autoencoder() #build empty model with the right architecure

    path_folder = os.path.dirname(__file__)#Get Current directory Path File
    model.load_weights(os.path.join(path_folder,"MODEL_WEIGHTS_FILE.h5")) #load weights only from h5 file

    return model

model = load_model_cache()

## Image Loader
uploaded_file = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])
res = None
image=None

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    imgArray = np.array(image).reshape(28,28,1) /255. #convert/resize and reshape
    if not imgArray.sum() >0:
        image = None
        st.write("Invalid Image")

    if st.button('Predict'):
        # Send to API
        if image is not None:

            caption = model.predict(image)
            st.write("Image caption:")
            st.write(caption)




            #----------------------------------------
