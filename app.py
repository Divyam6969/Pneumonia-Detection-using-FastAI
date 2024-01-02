from flask import Flask, render_template, request
from fastai.vision.learner import load_learner
from fastai.vision.core import PILImage
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)

# Load the exported learner
path = Path()
learn_inf = load_learner(path/'export.pkl')

# Dictionary to map model predictions to descriptions and risk factors
class_labels = {
    'VIRAL': {
        'label': 'Viral Pneumonia',
        'description': 'Viral pneumonia is an infection of the lungs caused by a virus. It can result in inflammation of the lung tissue and lead to symptoms such as cough, fever, difficulty breathing, and fatigue.',
        'risk_factors': 'Common viruses that can cause pneumonia include influenza (flu), respiratory syncytial virus (RSV), and the coronavirus. Other risk factors include age (young children and older adults are more vulnerable), weakened immune system, and underlying health conditions.'
    },
    'BACTERIAL': {
        'label': 'Bacterial Pneumonia',
        'description': 'Bacterial pneumonia is a lung infection caused by bacteria. It can cause inflammation in the air sacs of the lungs and lead to symptoms such as cough with phlegm, chest pain, high fever, and shortness of breath.',
        'risk_factors': 'Bacteria, such as Streptococcus pneumoniae, Haemophilus influenzae, and Mycoplasma pneumoniae, are common culprits. Risk factors include age (young children and older adults), weakened immune system, chronic lung diseases, smoking, and recent respiratory infections.'
    },
    'NORMAL': {
        'label': 'Normal',
        'description': 'Normal condition refers to the absence of pneumonia or other significant lung infections. The respiratory system functions without signs of inflammation or infection, and individuals experience normal breathing patterns and overall good health.',
        'risk_factors': 'While there are no specific risk factors for a normal lung condition, maintaining good respiratory hygiene, avoiding exposure to harmful pollutants, and adopting a healthy lifestyle contribute to lung health.'
    }
}


def predict_skin_cancer(img_path):
    img = PILImage.create(img_path)
    predicted_label, _, _ = learn_inf.predict(img)
    return predicted_label

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + secure_filename(img.filename)

        # Ensure the 'static' folder exists
        static_folder = Path("static")
        static_folder.mkdir(exist_ok=True)

        img.save(img_path)
        prediction = predict_skin_cancer(img_path)

        # Map the model prediction to the description and risk factors
        result = class_labels.get(prediction, {'label': 'Unknown', 'description': '', 'risk_factors': ''})

    return render_template("index.html", prediction=result['label'], description=result['description'], risk_factors=result['risk_factors'], img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
