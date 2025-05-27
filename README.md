# 🌾 Tech Kissan – Smart Farming Assistant 🚜

**Tech Kissan** is a smart farming web application developed as a Final Year Project to empower farmers with modern AI-driven tools. It combines crop recommendation, plant disease detection, review sentiment analysis, and a marketplace for agricultural products all in one platform.

---

# 🔑 Features

### 1. 🌱 Crop Recommendation System
- **Functionality**: Suggests the most suitable crop to grow based on user-input soil and environmental conditions.
- **Model Used**: Random Forest Classifier
- **Accuracy**: 99%
- **Input**: Soil parameters (Nitrogen, Phosphorus, Potassium, pH, Temperature, Humidity, Rainfall)
- **Output**: Recommended crop to cultivate

---

### 2. 🦠 Plant Disease Detection
- **Functionality**: Predicts whether a plant leaf is healthy or diseased from an uploaded image, and provides treatment suggestions if diseased.
- **Model Used**: ResNet50 (Deep Learning)
- **Accuracy**: 99%
- **Input**: Image of the plant leaf
- **Output**: Health status and suggested solutions

---

### 3. 💬 Review Sentiment Analyzer
- **Functionality**: Analyzes customer reviews for agricultural products and determines whether the sentiment is positive or negative.
- **Model Used**: Multinomial Naive Bayes
- **Accuracy**: 86%
- **Input**: Text review
- **Output**: Sentiment (Positive / Negative)

---

### 4. 🛒 Marketplace
- **Functionality**: A user-friendly interface for buying and browsing agricultural products.
- **Frontend Tech Stack**: HTML, CSS, JavaScript
- **Backend Integration**: Flask

---

## 🧠 Tech Stack

**Machine Learning & Deep Learning**
- `pandas`, `numpy`, `scikit-learn`, `joblib`
- `torch`, `torchvision`, `Pillow`

**Web Framework**
- `Flask`, `Jinja2`, `Werkzeug`, `MarkupSafe`

**Utilities**
- `requests`, `blinker`, `click`, `colorama`

**Frontend**
- `HTML5`, `CSS3`, `JavaScript`


---


---

# 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/tech-kissan.git
   cd tech-kissan

2. **Install Dependencies**
   ```
   pip install -r requirements.txt

3. **Run the Flask app**
   ```
   python app.py

4. **Access the app**
   ```
   http://127.0.0.1:5000/


