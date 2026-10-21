# 🛡️ PhishGuard — SMS Spam & Suspicious Message Analyzer

**PhishGuard** is a machine-learning-powered web application that analyzes text messages to identify spam and suspicious content. It combines a TF-IDF and Logistic Regression classification pipeline with rule-based risk indicators to help users recognize potentially harmful messages.

🔗 **Live Demo:** [PhishGuard on Streamlit](https://phishing-sms-and-mails.streamlit.app/)

💻 **Source Code:** [GitHub Repository](https://github.com/ak487max/Phishing-SMS-and-Mails)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Machine Learning Model](#-machine-learning-model)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Overview

Spam messages can contain misleading offers, suspicious links, requests for sensitive information, and attempts to manipulate users into taking risky actions.

PhishGuard provides an interactive interface where users can submit a message and receive a machine-learning prediction alongside additional rule-based indicators.

The project demonstrates the practical application of natural language processing (NLP), supervised machine learning, model evaluation, and interactive web application development.

## ✨ Features

- **SMS Classification:** Classifies messages as legitimate (ham) or spam using a trained machine learning pipeline.
- **Risk Indicators:** Identifies suspicious URLs, urgency-related language, sensitive-information requests, prize claims, payment requests, and possible impersonation.
- **Risk Analysis:** Generates a heuristic risk score from detected indicators to help users assess suspicious messages.
- **Interactive Dashboard:** Presents model evaluation metrics and classification performance.
- **Visual Analytics:** Uses interactive charts to display risk-related information and model results.
- **Message Inspection:** Highlights potentially suspicious content and extracts URLs for inspection.
- **Scan History:** Maintains a session-based history of analyzed messages.
- **CSV Export:** Allows users to export scan history where supported by the interface.
- **Streamlit Web Interface:** Provides an accessible interface without requiring users to run the model manually.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and model development |
| Scikit-learn | Machine learning and model evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | SMS spam classification |
| Pandas | Data handling and analysis |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Plotly | Interactive data visualizations |
| Matplotlib | Evaluation visualizations |

## 🤖 Machine Learning Model

PhishGuard uses a supervised machine learning pipeline trained on the SMS Spam Collection dataset.

### Workflow

1. **Data Preparation:** Loads and prepares labeled SMS messages.
2. **Feature Extraction:** Converts text into numerical features using TF-IDF, including unigram and bigram features.
3. **Classification:** Uses Logistic Regression with balanced class weights to classify messages.
4. **Evaluation:** Measures model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
5. **Deployment:** Loads the saved model into a Streamlit application for interactive predictions.

### Pipeline

`SMS Dataset → Text Preprocessing → TF-IDF → Logistic Regression → Prediction → Risk Indicators → Streamlit UI`

## 📊 Model Performance

The model achieved the following results on a held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 97.58% |
| Spam Precision | 90% |
| Spam Recall | 92% |
| Spam F1-Score | 91% |
| Test Samples | 1,034 |

These results represent performance on the evaluation dataset. They do not guarantee the same performance on unseen messages in real-world conditions.

## 📂 Project Structure

```text
Phishing-SMS-and-Mails/
│
├── app.py
├── dashboard.py
├── train_model.py
├── requirements.txt
├── evaluation_metrics.json
│
├── data/
│   └── SMSSpamCollection
│
└── models/
    └── message_classifier.joblib
```

*The structure above reflects the expected project layout; adjust it if your repository uses different filenames or folders.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or another compatible Python version
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ak487max/Phishing-SMS-and-Mails.git
```

### 2. Navigate to the project directory

```bash
cd Phishing-SMS-and-Mails
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run app.py
```

The application should open in your browser, usually at `http://localhost:8501`.

**Note:** The trained model and evaluation files must be available at the paths expected by the application. If they are not included in the repository, follow the project's training workflow to generate them before launching the app.

## 🖥️ Usage

1. Open the [live PhishGuard application](https://phishing-sms-and-mails.streamlit.app/).
2. Enter or paste the SMS message you want to analyze.
3. Run the analysis.
4. Review the predicted message category and detected risk indicators.
5. Explore the dashboard to understand model performance.
6. Export your scan history if the export option is available.

## ⚠️ Limitations

- The classification model is trained on SMS spam data and may not generalize to all phishing emails or newly emerging scams.
- Risk indicators are rule-based heuristics and should not be interpreted as definitive proof of malicious intent.
- Extracting a URL does not verify its reputation or establish whether a website is safe.
- A high model accuracy does not eliminate false positives or false negatives.
- PhishGuard should be used as an educational and decision-support tool, not as a replacement for security software or independent verification.

**Security reminder:** Never enter passwords, OTPs, banking credentials, or other sensitive information into suspicious links.

## 🔮 Future Improvements

- Extend the training dataset with labeled phishing emails and more diverse SMS examples.
- Add email-specific classification and analysis.
- Integrate trusted URL reputation services.
- Improve model explainability with feature-level insights.
- Evaluate additional NLP models and compare their performance.
- Add more comprehensive evaluation on unseen and real-world datasets.
- Improve deployment monitoring and automated testing.

## 👩‍💻 Author

**Akansha Mishra**

B.Tech — Computer Science and Engineering (AI & ML)

- **GitHub:** [ak487max](https://github.com/ak487max)
- **Project Repository:** [Phishing-SMS-and-Mails](https://github.com/ak487max/Phishing-SMS-and-Mails)
- **Live Demo:** [PhishGuard](https://phishing-sms-and-mails.streamlit.app/)

---

*Built to explore machine learning, NLP, and practical message security analysis.*

