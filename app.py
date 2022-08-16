import sys
import logging
import joblib
from PIL import Image
import numpy as np
import  cv2
from keras.applications.vgg16 import preprocess_input
import json
from flask import Flask, request, Response


application = Flask(__name__)
logger = logging.getLogger("beer_project_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@application.route('/beer_guess', methods=['POST'])
def guess_beer():
    
    response = None
    try:
        logger.info("Receiving Image")
        image = request.files['image']
        label = compute_predictions(image)

        logger.info("Generating cupom messages")
        cupom_messages = generate_cumpom_messages(label)
        cupons_description = cupom_messages[0]
        cupons_tag = cupom_messages[1]

        response = Response(content_type='application/json', status = 200, response = json.dumps({"Label": label, "Message": cupons_description, "Tags": cupons_tag}))
    except BaseException as error:
        error_content = {"Error": str(error)}
        logger.error("Error", json.dumps(error_content))
        response = Response(content_type='application/json', status = 500, response = json.dumps(error_content))
    return response


def preprocess_image(image):
    logger.info("Preprocessing image")
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
    logger.info("Decoding predictions")

    items = np.argsort(pred[0])[-5:][::-1]
    classes_dict = get_dict_labels()
    label = list(map(lambda x: classes_dict[x], items))[0]
    return label

def compute_predictions(image):
    logger.info("Model load")
    model = joblib.load('model_beers.pickle')
    image_processed =  preprocess_image(image)
    predictions = model.predict(image_processed)
    final_label = decode_predictions_custom(predictions)
    return final_label

def generate_cumpom_messages(beer):
    cupom_list = ['10% na primeira compra', '5% na compra da sua cerveja', 
    'Na compra da primeira, leve a segunda com 10% de desconto']
    tags = ["CERVEJA10", "CERVEJA5", "LEVE1PAGUEMENOS"]

    number_of_cupoms = np.random.randint(1,4)
    cupom_chosed = np.random.choice(cupom_list, number_of_cupoms, replace=False)
    cupom_str = "Cupons: "
    tags_str = ""

    for i in range(number_of_cupoms):
        cupom_str = f"{cupom_str} {cupom_chosed[i]}, "
        tags_str = f"{tags_str}{tags[i]}, "

    return [cupom_str, tags_str]
    
if __name__ == '__main__':
     application.run(host='0.0.0.0', port=8080)