import streamlit as st
import numpy as np
from PIL import Image
import os
from model import model_builder, predict_caption, image_to_features, image_to_array #links with model building .py file
from dotenv import load_dotenv, find_dotenv
import openai
from tensorflow.image import resize

import time
import requests


env_path = find_dotenv()
load_dotenv(env_path)

# retrieve access password (opt: add a login)
app_password = os.environ.get("APP_PASSWORD")
# retrieve access to text to speech
text_to_speech = os.environ.get("tts_API")
# retrieve api key for GPT3
openai.api_key = os.getenv('OPENAI_KEY')

st.sidebar.title("#❤️ Robo-Romeo ❤️")

# ask for password
password = st.sidebar.text_input("Enter a password", type="password")
if password == app_password:

    #st.title("Robo-Romeo❤️")
    st.text("Upload your image - ❤️ Robo-Romeo ❤️ will provide you with a caption and romantic poetry")


    @st.cache(allow_output_mutation=True) #cache the model at first loading
    def load_model_cache():

        model = model_builder() #build empty model with the right architecure

    #Get Current directory Path File
        model.load_weights("model_run_30k_weights.h5") #load weights only from h5 file

        return model

    model = load_model_cache()

    ## Image Loader
    uploaded_file = st.sidebar.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])
    res = None
    image=None


    if uploaded_file:
        image = Image.open(uploaded_file)
        #imgArray = np.array(image) #convert/resize and reshape
        #imgArray = resize(imgArray,(256,256))
        #imgArray = np.expand_dims(imgArray, axis=0)

        imgArray = image_to_array(uploaded_file)

        predict_button = st.sidebar.button('wherefore art thou Romeo?')
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

                #Text to Speach API Request

                # first request to get unique 'uuid' of the inputed text
                url = "https://api.uberduck.ai/speak"
                payload = {
                    "voice": "c-3po",
                    "pace": 1,
                }
                payload["speech"] = response.choices[0].text

                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }

                headers["Authorization"] = text_to_speech

                r = requests.post(url, json=payload, headers=headers)

                time.sleep(5)

                # second request to get the link for WAV file
                url_2 = f"https://api.uberduck.ai/speak-status?uuid={r.json()['uuid']}"
                headers_2 = {"Accept": "application/json"}

                r_2 = requests.get(url_2 ,headers=headers_2)

                audio_file = r_2.json()['path']

                # display the audio
                st.markdown(f"Play The Audio:")
                st.audio(audio_file, format="audio/wav", start_time=0)
