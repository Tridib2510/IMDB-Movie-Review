# IMDB Movie Review Sentiment Analysis

This project implements a sentiment analysis model to classify IMDb movie reviews as positive or negative. Utilizing a Simple Recurrent Neural Network (RNN) architecture, the model processes textual data to predict sentiments effectively.

## 🧠 Project Overview

The goal of this project is to develop a machine learning model capable of analyzing movie reviews from IMDb and determining whether the sentiment expressed is positive or negative. The model is built using TensorFlow and Keras, leveraging a Simple RNN for sequence processing.

## ⚙️ Technologies Used

- **Python 3.13.8**: Programming language used for development.
- **TensorFlow 2.20.0**: Deep learning framework for building and training the RNN model.
- **Keras**: High-level neural networks API, running on top of TensorFlow.
- **Streamlit**: Framework for creating interactive web applications to deploy the model.
- **NumPy & Pandas**: Libraries for data manipulation and analysis.
- **Matplotlib & Seaborn**: Libraries for data visualization.

## 📁 Project Structure

IMDB-Movie-Review/
│
├── main.py # Streamlit application for model inference
├── prediction.ipynb # Jupyter notebook for model training and evaluation
├── requirements.txt # Python dependencies
├── simple_rnn_imdb.h5 # Trained RNN model weights
└── .gitignore # Git ignore file

bash
Copy code

## 📦 Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Tridib2510/IMDB-Movie-Review.git
   cd IMDB-Movie-Review
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
🚀 Usage
Running the Streamlit App
To launch the interactive web application:

bash
Copy code
streamlit run main.py
This will start a local server, and you can access the application in your web browser.

Training the Model
For training the model:

Open prediction.ipynb in a Jupyter notebook environment.

Follow the steps outlined in the notebook to preprocess the data, build the model, and train it.

Model Inference
Once the model is trained:

The main.py script loads the pre-trained model (simple_rnn_imdb.h5) and provides an interface to input movie reviews for sentiment prediction.

📊 Dataset
The project utilizes the IMDb movie reviews dataset, which contains 50,000 reviews labeled as positive or negative. This dataset is commonly used for training sentiment analysis models.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

pgsql
Copy code

You can copy and paste this directly into your `README.md`.  

If you want, I can also **add badges for Python version, Streamlit, and TensorFlow** to make it look more professional. Do you want me to do that?






