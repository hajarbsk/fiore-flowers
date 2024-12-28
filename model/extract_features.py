import os
import numpy as np
from resnet50 import extract_features

# Chemin vers vos images
BASE_DIR = '../static/test/'

# Dictionnaire pour stocker les caractéristiques
features_db = {}

for img_name in os.listdir(BASE_DIR):
    img_path = os.path.join(BASE_DIR, img_name)
    features = extract_features(img_path)
    features_db[img_name] = features

# Sauvegarder les caractéristiques dans un fichier
np.save('features_db.npy', features_db)
print("Extraction des caractéristiques terminée.")
