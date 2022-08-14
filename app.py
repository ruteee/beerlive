import joblib
import logging
from PIL import Image
import json
import numpy as np
import  cv2
from keras.applications.vgg16 import preprocess_input
from flask import Flask, request, Response


application = Flask(__name__)
loggerr = logging.getLogger()

@application.route('/beer_guess', methods=['POST'])
def guess_beer():
    
    response = None
    try:
        loggerr.info("Receiving Image")
        image = request.files['image']
        label = compute_predictions(image)
        response = Response(content_type='application/json', status = 200, response = json.dumps({"Beer": label}))
    except BaseException as error:
        error_content = {"Error": str(error)}
        logging.error("Erro on receiving request ", json.dumps(error_content))
        response = Response(content_type='application/json', status = 400, response = json.dumps(error_content))
    return response


def preprocess_image(image):
    image_processed = Image.open(image)
    image_processed = np.array(image_processed)
    image_processed = cv2.resize(src= image_processed, dsize = (224, 224))
    image_processed = image_processed.reshape((1, image_processed.shape[0], image_processed.shape[1], image_processed.shape[2]))
    image_processed = preprocess_input(image_processed)
    return image_processed


def get_dict_labels():
    mapping_labels =  np.array(['antartica_lata', 'antartica_longneck', 'bohemia_lata',
       'bohemia_longneck', 'brahma_lata', 'brahma_longneck',
       'budweiser_lata', 'budweiser_longneck', 'colorado_garrafa',
       'colorado_lata', 'corona', 'skol_lata', 'skol_litrao',
       'stella_artois_longneck'])

    classes_dict = {}
    for i,classe in enumerate(mapping_labels):
        classes_dict[i] = classe
    return classes_dict

def decode_predictions_custom(pred):
    items = np.argsort(pred[0])[-5:][::-1]
    classes_dict = get_dict_labels()
    label = list(map(lambda x: classes_dict[x], items))[0]
    return label

def compute_predictions(image):
    model = joblib.load('model_beers.pickle')
    image_processed =  preprocess_image(image)
    predictions = model.predict(image_processed)
    final_label = decode_predictions_custom(predictions)
    return final_label


if __name__ == '__main__':
     application.run(host='0.0.0.0', port=8080)