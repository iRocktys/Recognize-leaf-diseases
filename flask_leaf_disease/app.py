from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Criar pasta de uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carregar modelo treinado
model_path = "model/plant_disease_model.h5"
model = tf.keras.models.load_model(model_path)

# Mapeamento de classes
class_labels = ["Bacterial Spot", "Healthy"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Nenhum arquivo enviado", 400

    file = request.files['file']

    if file.filename == '':
        return "Nenhum arquivo selecionado", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Processar imagem e fazer previsão
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return f"A folha foi classificada como: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
