from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Criar diretório de uploads se não existir
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Carregar modelo treinado
model = tf.keras.models.load_model('model/plant_disease_model.h5')

# Mapeamento das classes formatadas
classes = {
    'Pepper_bell__Bacterial_spot': 'Pimentão - Mancha Bacteriana',
    'Pepper_bell__healthy': 'Pimentão - Saudável',
    'Potato__Early_blight': 'Batata - Requeima Precoce',
    'Potato__healthy': 'Batata - Saudável',
    'Potato__Late_blight': 'Batata - Requeima Tardia',
    'Tomato__Bacterial_spot': 'Tomate - Mancha Bacteriana',
    'Tomato__Target_Spot': 'Tomate - Mancha Alvo',
    'Tomato__mosaic_virus': 'Tomate - Vírus do Mosaico',
    'Tomato__YellowLeaf_Curl_Virus': 'Tomate - Vírus do Enrolamento Amarelo',
    'Tomato_Early_blight': 'Tomate - Requeima Precoce',
    'Tomato_healthy': 'Tomate - Saudável',
    'Tomato_Late_blight': 'Tomate - Requeima Tardia',
    'Tomato_Leaf_Mold': 'Tomate - Mofo das Folhas',
    'Tomato_Septoria_leaf_spot': 'Tomate - Mancha Septoriana',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomate - Ácaro de Duas Manchas'
}

# Histórico de uploads
history = []

# Função para processar imagem e classificar
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = list(classes.keys())[class_index]
    return classes[class_name]

@app.route('/', methods=['GET', 'POST'])
def index():
    global history
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            result = classify_image(file_path)
            history.append({'image': file_path, 'result': result})
            history = history[-3:]  # Manter apenas os 3 últimos
    return render_template('index.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)
