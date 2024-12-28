from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Charger le modèle ResNet50 pré-entraîné
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    # Charger l'image et la convertir en tableau
    
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extraire les caractéristiques
    features = model.predict(img_array)
    return features.flatten()
