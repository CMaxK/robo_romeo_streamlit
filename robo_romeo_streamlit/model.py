import numpy as np
import pickle
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Embedding, Add, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
def image_to_array(path):
    loaded_img = image.load_img(path, target_size=(256,256,3))
    img_array = image.img_to_array(loaded_img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
def cnn_builder():
    cnn_model = EfficientNetB0(include_top=False, weights='imagenet',
                           input_shape= (256,256,3))
    return cnn_model
def image_to_features(path):
    img_array = image_to_array(path)
    features = cnn_builder().predict(img_array)
    return features
def model_builder():
    cap_len = 25
    vocab_size = 19758
    embed_dim = 1024
    inputs2  = Input(shape=(cap_len,),name="captions")
    embed_layer = Embedding(vocab_size+3, embed_dim, mask_zero=True)(inputs2)
    input_encoded = Input(shape=(8,8,1280),name="images_encoded")
    pooling = GlobalAveragePooling2D()(input_encoded)
    cnn_dense = Dense(embed_dim, activation='relu')(pooling)
    combine = Add()([embed_layer,cnn_dense])
    lstm_layer = LSTM(embed_dim)(combine)
    decoder = Dense(2048, activation='relu')(lstm_layer)
    outputs = Dense(vocab_size+1, activation='softmax')(decoder)
    lstm_model = Model(inputs=[input_encoded, inputs2], outputs=outputs)
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return lstm_model
def predict_caption(model,img_encoded):
    cap_len = 25
    index_file ='word_index.pkl'
    with open(index_file,'rb') as handle:
        word_index = pickle.load(handle)
    inputs_seq = [2]
    endsequence_token = word_index['endsequence']
    for i in range(cap_len):
        inputs_seq_model = pad_sequences([inputs_seq],padding='post',maxlen=25)
        y_pre = model.predict([img_encoded,inputs_seq_model])
        next_word = y_pre.argmax()
        if next_word == endsequence_token:
            break
        inputs_seq.append(next_word)
    sentence = []
    for number in inputs_seq:
        sentence.append(list(word_index.keys())[list(word_index.values()).index(number)])
    sentence = sentence [1:]
    prediction = ' '.join(word for word in sentence)
    return prediction
if __name__ == '__main__':
    lstm_model = model_builder()
    lstm_model.load_weights('model_run_30k_weights.h5')
    img_encoded = image_to_features('IMG_9640.jpg')
    caption = predict_caption(lstm_model, img_encoded)
    print(caption)
