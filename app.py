import streamlit as st
import numpy as np
from PIL import Image
import os
from model import model_builder, predict_caption, image_to_features, image_to_array #links with model building .py file
from dotenv import load_dotenv, find_dotenv
import openai
from tensorflow.image import resize

env_path = find_dotenv()
load_dotenv(env_path)

# retrieve access password (opt: add a login)
app_password = os.environ.get("APP_PASSWORD")
# retrieve api key for GPT3
openai.api_key = os.getenv('OPENAI_KEY')

# ask for password
password = st.text_input("Enter a password", type="password")
if password == app_password:

    st.title("Robo-Romeo❤️")
    st.markdown("Upload your image - Robo-Romeo will provide you with a caption and romantic poetry")


    @st.cache(allow_output_mutation=True) #cache the model at first loading
    def load_model_cache():

        model = model_builder() #build empty model with the right architecure

    #Get Current directory Path File
        model.load_weights("model_run_30k_weights.h5") #load weights only from h5 file

        return model

    model = load_model_cache()

    ## Image Loader
    uploaded_file = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])
    res = None
    image=None


    if uploaded_file:
        image = Image.open(uploaded_file)
        #imgArray = np.array(image) #convert/resize and reshape
        #imgArray = resize(imgArray,(256,256))
        #imgArray = np.expand_dims(imgArray, axis=0)

        imgArray = image_to_array(uploaded_file)

        predict_button = st.button('Predict')
        placeholder = st.empty()

        if not imgArray.sum() >0:
            image = None
            st.write("Invalid Image")
        else:
            placeholder.image(image)

        if predict_button:
            # Send to API

            if image is not None:

                img_encoded = image_to_features(imgArray)
                caption = predict_caption(model, img_encoded)
                placeholder.image(image,caption=caption)

                # GPT3 function that will return romantic poem. takes predicted caption as input
                def gpt3(prompt=f"write a love poem about {caption}:", engine='text-davinci-002',
                        temperature=0.7,top_p=1, max_tokens=256,
                        frequency_penalty=0, presence_penalty=0):
                    response = openai.Completion.create(engine=engine,
                                                    prompt=prompt,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens,
                                                    top_p=top_p,
                                                    frequency_penalty=frequency_penalty,
                                                    presence_penalty=presence_penalty)
                    return response

                # instanciate gpt3 function with previously confirmed parameters
                response = gpt3()

                # prints poem
                st.text(response.choices[0].text)

                #----------------
