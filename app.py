import numpy as np
from flask import Flask, request, render_template, url_for
from model.resnet50 import extract_features
from model.utils import cosine_similarity

app = Flask(__name__)

# Charger la base de données des caractéristiques
features_db = np.load('model/features_db.npy', allow_pickle=True).item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "Aucune image téléchargée.", 400

    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné.", 400

    # Sauvegarder l'image téléchargée
    upload_path = f"static/uploads/{file.filename}"
    file.save(upload_path)

    # Extraire les caractéristiques de l'image téléchargée
    uploaded_features = extract_features(upload_path)

    # Comparer avec la base de données
    similarities = {}
    for img_name, features in features_db.items():
        similarities[img_name] = cosine_similarity(uploaded_features, features)

    # Trier les résultats par similarité
    similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Renvoyer les 5 images les plus similaires
    results = [img for img, score in similar_images[:5]]

    return render_template('results.html', uploaded_image=upload_path, results=results)


if __name__ == '__main__':
    app.run(debug=True)