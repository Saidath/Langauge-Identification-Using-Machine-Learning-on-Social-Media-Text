# Language Identification Using Machine Learning on Social Media Text

This project is a web application that detects the language of input sentences, identifying whether the language is English or Hindi. Built with **Flask** as the backend framework, it uses a pre-trained LSTM model to classify the input language. The frontend is styled with **Bootstrap** to provide a responsive and visually appealing user experience.

## Project Overview

- **Frontend:** HTML, CSS (with Bootstrap), and JavaScript
- **Backend:** Flask
- **Machine Learning Model:** LSTM model for language detection
- **Dataset:** `english_words.xlsx` file containing lists of English and Hindi words for reference

## Features

- Predicts if a given sentence is in English or Hindi
- Displays prediction accuracy and processing time for each prediction
- Easy-to-use web interface

## Prerequisites

To run this project, ensure you have:

- **Python 3.7+**
- **TensorFlow**
- **Flask**
- **Pandas**
- **OpenCV**
- An `english_words.xlsx` file containing columns for English and Hindi words

## Getting Started

1. **Clone the repository:**

   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name


2. **Install dependencies:**

   pip install -r requirements.txt


3. **Prepare Dataset:**
   - Ensure you have an `english_words.xlsx` file in the root directory, with columns:
     - `English`: containing English words
     - `Hindi`: containing Hindi words

4. **Train the LSTM Model:**
   - Modify `app.py` to include your model training code, or load the pre-trained model weights file `model_weights.h5`.

5. **Run the Application:**

   python app.py

   The application should start on `http://127.0.0.1:5000/`.

## Project Structure

```
.
├── app.py               # Flask backend with LSTM model integration
├── templates
│   └── index.html       # Frontend HTML file
├── static
│   ├── css
│   │   └── style.css    # Custom CSS styling for the app
├── english_words.xlsx   # Excel file containing English and Hindi words
├── model_weights.h5     # Pre-trained model weights
└── README.md            # Project documentation
```

## Usage

1. Open the web application in your browser at `http://127.0.0.1:5000/`.
2. Enter a sentence in the input field and click **Submit**.
3. The predicted language will be displayed along with accuracy and processing time.


## Acknowledgements

- **TensorFlow** for model building
- **Flask** for backend
- **Bootstrap** for responsive styling
