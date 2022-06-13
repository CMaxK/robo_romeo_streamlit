import streamlit as st
import numpy as np
from PIL import Image
import os
from model import build_model #links with model building .py file
from dotenv import load_dotenv, find_dotenv
import openai

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

        model = build_model() #build empty model with the right architecure

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
                st.caption(caption)

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
                st.text(response)

                #----------------

else:

    "incorrect password ❌"
