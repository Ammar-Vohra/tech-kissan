# Importing essential libraries and modules
from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
import joblib  # Import joblib for model loading
from utils.disease import disease_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import base64
from io import BytesIO
from PIL import Image

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------


# Loading plant disease classification model
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()



# Load the crop recommendation model, scaler, and encoder using joblib
crop_recommendation_model = joblib.load('models/random_forest_model.pkl')
encoder = joblib.load('models/encoder.pkl')
scaler = joblib.load('models/scaler.pkl')


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# =========================================================================================

app = Flask(__name__)

@app.route('/')
@app.route("/home")
def home():
    return render_template("home.html", title="Home")


@app.route("/plant_disease")
def plant_disease():
    return render_template("plant_disease.html", title="Plant Disease")


@app.route("/review_analyzer")
def review_analyzer():
    return render_template("review_analyzer.html", title="Review Analyzer")

@app.route("/market_place")
def market_place():
    return render_template("market_place.html", title="Market Place")

@app.route("/about")
def about():
    return render_template("about.html", title="About Us")

@app.route("/crop_recommendation")
def crop_recommendation():
    return render_template("crop_recommendation.html", title="Crop Recommendation")

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Kisan Tech - Crop Recommendation'

    if request.method == 'POST':
        # Retrieve form data
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        temperature = float(request.form['temperature'])  # Take temperature as input
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare data for prediction
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Apply scaling to the input data
        scaled_data = scaler.transform(data)

        # Make prediction using the trained model
        my_prediction = crop_recommendation_model.predict(scaled_data)

        # Inverse transform the prediction to get the actual crop label
        final_prediction = encoder.inverse_transform([my_prediction])[0]

        # Render the prediction result on the result page
        return render_template('crop-result.html', prediction=final_prediction, title=title)



@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    title = 'Kisan Tech - Plant Disease Prediction'

    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return render_template('plant_disease.html', error="No file uploaded!", title=title)

        file = request.files['image']
        if file.filename == '':
            return render_template('plant_disease.html', error="No file selected!", title=title)

        if file:
            try:
                # Open the image in memory using PIL
                img = Image.open(file.stream)
                
                # Convert image to base64 to display it in HTML
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

                # Predict the disease
                prediction = predict_image(img_byte_arr)  # Pass image bytes for prediction
                solution = disease_dic.get(prediction, "No details available for this disease.")

                # Render result page with prediction and solution
                return render_template('plant_disease_result.html', 
                                       prediction=prediction, 
                                       solution=Markup(solution), 
                                       img_data=img_base64, 
                                       title=title)
            except Exception as e:
                return render_template('plant_disease.html', error=f"Error during prediction: {str(e)}", title=title)

    return redirect('/plant_disease')


# Load the vectorizer and model
with open('models/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route("/review_prediction", methods=["POST"])
def review_prediction():
    title = 'Kisan Tech - Review Analyzer'

    if request.method == 'POST':
        review = request.form['review']
        
        if not review.strip():
            return render_template('review_analyzer.html', sentiment="Please enter a valid review.")

        # Vectorize the review
        review_vectorized = vectorizer.transform([review])
        
        # Predict the sentiment and get probabilities
        prediction = model.predict(review_vectorized)[0]
        
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(review_vectorized)[0]
            confidence = max(probas) * 100  # Convert to %
        else:
            confidence = "N/A"  # If model doesn't support predict_proba

        # Sentiment label
        sentiment = 'Positive' if prediction == 2 else 'Negative'

        return render_template('review_analyzer.html', sentiment=sentiment, score=round(confidence, 2), title=title)

    return render_template('home.html', sentiment=None)








if __name__ == "__main__":
    app.run(debug=True)

