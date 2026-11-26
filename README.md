
<div align="center">

# 🚀 ReturnIQ

### 🛍️ E-Commerce Returns Intelligence Platform

**Every return has a reason. Find it.**

<img src="docs/images/landing-page.png" width="100%" alt="ReturnIQ Landing Page"/>

<br/>

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Explore_ReturnIQ-294B3C?style=for-the-badge)](https://return-iq.vercel.app)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://return-iq.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://returniq-backend-90p6.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)

</div>

---

## 💡 Overview

ReturnIQ is an AI-powered e-commerce returns intelligence platform that transforms unstructured customer return comments into actionable business insights using Natural Language Processing (NLP) and Machine Learning.

E-commerce businesses receive thousands of product return comments. These comments contain valuable information about product sizing, damaged items, incorrect products, and delivery issues.

ReturnIQ automatically analyzes and classifies these comments, helping businesses understand customer dissatisfaction, identify recurring issues, and make better operational decisions.

### 🎯 Return Categories

| Category | Description |
|---|---|
| 📏 Sizing | Product size or fit did not meet expectations |
| 📦 Damaged Item | Product arrived damaged or defective |
| 🔄 Wrong Product | Customer received an incorrect product |
| 🚚 Late Delivery | Product arrived later than expected |

---

## ✨ Key Features

- 🤖 **AI-powered classification** – Automatically identifies customer return reasons.
- 🧠 **Two ML models** – TF-IDF + Logistic Regression and fine-tuned DistilBERT.
- 🔍 **Prediction Studio** – Classifies individual customer comments.
- 📊 **Probability distribution** – Displays scores across all four categories.
- 📁 **Batch Intelligence** – Supports batch-oriented return comment analysis.
- 🗂️ **Dataset Explorer** – Explore the classification dataset.
- 🧪 **Model Laboratory** – Compare machine learning models.
- 📈 **Confusion Matrix** – Visualize actual and predicted classifications.
- 🗄️ **PostgreSQL integration** – Database-backed application.
- ⚡ **Fast inference** – Lightweight TF-IDF model for production deployment.

---

## 🖥️ Application Screenshots

### 🏠 Landing Page

The landing page introduces ReturnIQ and its customer returns intelligence workflow.

<img src="docs/images/landing-page.png" width="100%" alt="ReturnIQ Landing Page"/>

### 🔮 Prediction Studio

Classify individual customer return comments and inspect the predicted category and probability distribution.

<img src="docs/images/prediction-studio.png" width="100%" alt="ReturnIQ Prediction Studio"/>

### 🧪 Model Laboratory

Compare traditional machine learning with transformer-based NLP using the same test dataset.

<img src="docs/images/model-laboratory.png" width="100%" alt="ReturnIQ Model Laboratory"/>

### 📊 Confusion Matrix

Visualize classification performance for both models across the four return categories.

<img src="docs/images/confusion-matrix.png" width="100%" alt="TF-IDF and DistilBERT Confusion Matrices"/>

---

## 🧠 Machine Learning Models

ReturnIQ implements two different machine learning approaches for customer return reason classification.

### 1. TF-IDF + Logistic Regression

A traditional machine learning approach that converts customer comments into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).

The extracted word and bigram features are classified using Logistic Regression.

**Advantages:**
- Lightweight model architecture.
- Fast CPU inference.
- Interpretable feature weights.
- Suitable for resource-constrained deployment.

### 2. Fine-tuned DistilBERT

A transformer-based NLP model fine-tuned for the four return reason categories.

DistilBERT uses contextual token representations to understand the meaning of customer comments beyond individual word frequencies.

**Advantages:**
- Context-aware text representations.
- Better understanding of word relationships.
- Fine-tuned for the ReturnIQ classification task.
- Supports contextual text classification.

### ⚖️ Model Comparison

| Feature | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |
|---|---|---|
| Approach | Traditional ML | Transformer NLP |
| Text representation | Word and bigram TF-IDF | Contextual embeddings |
| Model architecture | Logistic Regression | DistilBERT |
| Inference | Lightweight | Computationally intensive |
| Interpretability | Feature weights | Contextual representations |
| Production deployment | Render | Local development |

---

## 📈 Model Evaluation

Both models were evaluated using the same **3,000-row synthetic test split**, containing 750 examples per category.

### Evaluation Metrics

| Metric | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |
|---|---:|---:|
| Accuracy | 100.0% | 100.0% |
| Macro Precision | 100.0% | 100.0% |
| Macro Recall | 100.0% | 100.0% |
| Macro F1-score | 100.0% | 100.0% |
| Weighted F1-score | 100.0% | 100.0% |

### Confusion Matrix Summary

| Actual Category | TF-IDF Correct | DistilBERT Correct |
|---|---:|---:|
| Sizing | 750 | 750 |
| Damaged Item | 750 | 750 |
| Wrong Product | 750 | 750 |
| Late Delivery | 750 | 750 |
| **Total** | **3,000** | **3,000** |

### Evaluation Note

The displayed results are development results obtained using synthetic data. They do not establish real-world model performance.

Further validation using representative, independently labelled customer return comments is necessary before using these predictions for actual business decisions.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Customer Return Comment"] --> B["React + Vite Frontend"]
    B --> C["FastAPI Backend"]
    C --> D{"Selected ML Model"}

    D --> E["TF-IDF + Logistic Regression"]
    D --> F["Fine-tuned DistilBERT"]

    E --> G["Predicted Return Category"]
    F --> G

    G --> H["Probability Distribution"]
    H --> B

    C --> I[("PostgreSQL Database")]
```

### 🔄 Prediction Workflow

1. Customer enters a return comment.
2. Frontend sends the comment to the FastAPI backend.
3. Backend processes the selected machine learning model.
4. Model predicts one of the four return categories.
5. Prediction confidence and class probabilities are returned.
6. Frontend displays the classification results.

---

## 🛠️ Technology Stack

### Frontend
- React
- TypeScript
- Vite
- HTML5
- CSS3

### Backend
- Python
- FastAPI
- SQLAlchemy
- Pydantic

### Machine Learning & NLP
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Hugging Face Transformers
- DistilBERT
- PyTorch

### Database
- PostgreSQL
- SQLite (local development, if configured)

### Tools & Deployment
- Git
- GitHub
- VS Code
- Vercel
- Render
- REST API

---

## 📁 Project Structure

```text
ReturnIQ/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   └── ...
│   │
│   ├── requirements.txt
│   └── .python-version
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── models/
│   ├── tfidf_returniq.joblib
│   │
│   └── distilbert/
│       └── final/
│
├── docs/
│   └── images/
│       ├── landing-page.png
│       ├── prediction-studio.png
│       ├── confusion-matrix.png
│       └── model-laboratory.png
│
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started

### 📌 Prerequisites

Install the following:

- Python 3.12
- Node.js and npm
- Git
- PostgreSQL (if using a local PostgreSQL database)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Lakshith-P/ReturnIQ.git

cd ReturnIQ
```

### 2️⃣ Backend Setup

Open a terminal:

```powershell
cd backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create a `.env` file inside the backend directory.

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/returniq
FRONTEND_ORIGIN=http://localhost:5173
MODEL_DIR=models/tfidf
```

Replace the database credentials with your own local PostgreSQL credentials.

**Never commit `.env` files, database passwords, or production secrets to GitHub.**

### 4️⃣ Start the Backend

```powershell
uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### 5️⃣ Frontend Setup

Open another terminal:

```powershell
cd frontend

npm install
```

Create a `.env` file inside the frontend directory:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start the frontend:

```powershell
npm run dev
```

Open the application:

```text
http://localhost:5173
```

### 6️⃣ Local Model Availability

The local development environment supports:

- TF-IDF + Logistic Regression
- Fine-tuned DistilBERT

Ensure the corresponding trained model artifacts and required ML dependencies are available before selecting a model.

---

## 🔌 API Documentation

ReturnIQ provides REST API endpoints through FastAPI.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Backend health and model availability |
| `/api/predict` | POST | Predict a customer return reason |

### Example Prediction Request

```json
{
  "text": "The shoes are too small for my feet",
  "model": "tfidf"
}
```

### Example Prediction Response

```json
{
  "predicted_category": "Sizing",
  "predicted_reason": "Product size or fit did not meet expectations",
  "confidence": 0.5576,
  "model_used": "TF-IDF + Logistic Regression",
  "demo_fallback": false
}
```

The actual prediction confidence and probability distribution depend on the customer comment and selected model.

---

## ☁️ Deployment

ReturnIQ uses separate hosting services for its frontend and backend.

### 🌐 Frontend – Vercel

Live application:

https://return-iq.vercel.app

### ⚙️ Backend – Render

Backend service:

https://returniq-backend-90p6.onrender.com

### 🚀 Production Model Availability

| Component | Environment | Status |
|---|---|---|
| React Frontend | Vercel | Deployed |
| FastAPI Backend | Render | Deployed |
| PostgreSQL | Render | Configured |
| TF-IDF + Logistic Regression | Render | Production |
| Fine-tuned DistilBERT | Localhost | Development |

The production backend uses the lightweight TF-IDF model because of the memory limitations of the Render Free instance.

Both models are available in the local development environment.

---

## 🔮 Future Improvements

- Validate classification using real-world customer return datasets.
- Improve model performance on ambiguous customer comments.
- Add advanced batch analytics and return trend visualization.
- Implement explainable AI for prediction-level insights.
- Explore optimized transformer inference for resource-constrained deployment.
- Introduce product-level return analysis.
- Generate automated operational recommendations.

---

## 🎯 Project Goal

ReturnIQ aims to transform customer return feedback into meaningful business intelligence.

By identifying recurring return reasons, e-commerce businesses can better understand customer dissatisfaction, improve product quality, optimize product descriptions, and enhance delivery experiences.

---

<div align="center">

## 🌿 ReturnIQ

### Customer Signal. Business Clarity.

**Turning customer feedback into actionable intelligence.**

[🌐 Live Demo](https://return-iq.vercel.app) · [💻 GitHub Repository](https://github.com/Lakshith-P/ReturnIQ)

Built with ❤️ using Machine Learning, NLP and Full-Stack Development.

</div>
