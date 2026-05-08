# Career Navigator AI 🎓

A multi-page Streamlit application demonstrating **Explainable AI** for career guidance. Students can explore how machine-learning models make career predictions and understand the reasoning behind them using SHAP, LIME, and LLM-powered interpretations.

## 🌟 Features

- **Multi-page Wizard**: Step-by-step flow — Dataset → Train → Predict → Explain → Interpret.
- **Multiple ML Models**: Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, SVM.
- **XAI Method Selection**: Choose between SHAP and LIME (or both), with configurable run counts.
- **Dynamic Datasets**: Pull career-guidance datasets directly from Kaggle.
- **LLM Interpretation**: "Why did the AI predict that?" — explained in plain English via Ollama.

## 📂 Project Structure

```
xai-app/
├── app.py                      # Entrypoint — st.navigation router
├── pages/                      # Multi-page Streamlit pages
│   ├── 1_home.py               #   XAI category selection
│   ├── 2_dataset.py            #   Dataset download & preview
│   ├── 3_train.py              #   Model selection & training
│   ├── 4_predict.py            #   Custom data input & prediction
│   ├── 5_explain.py            #   SHAP/LIME configuration & execution
│   ├── 6_interpret.py          #   LLM-powered explanation
│   ├── viz_1_data.py           #   Basic data visualization & analysis
│   ├── viz_2_train.py          #   Basic model training visualization
│   ├── viz_3_predict.py        #   Basic prediction visualization
│   ├── viz_4_explain.py        #   Basic explanation visualization
│   ├── viz_5_counselor.py      #   Counselor career guidance questions
│   ├── viz_adv_1_analyze.py    #   Advanced visual analytics with Altair
│   ├── viz_adv_2_transcribe.py #   VLM handwritten math review
│   ├── viz_adv_3_explain.py    #   Explain handwritten math with LLM
│   ├── viz_adv_4_review.py     #   Review math solution with LLM
│   ├── viz_adv_5_quiz.py       #   AI quiz master for math problems
│   └── viz_adv_6_interview.py  #   AI interview simulator for careers
├── core/                   # ML models, XAI engines, LLM integration
│   ├── models.py           #   Multi-model registry & training
│   ├── shap_engine.py      #   SHAP with auto-explainer routing
│   ├── lime_engine.py      #   LIME tabular explainer
│   ├── llm.py              #   Ollama LLM client
│   ├── vlm_engine.py       #   Qwen2.5-VL VLM client for math review
│   └── preprocessing.py    #   Data encoding & task detection
├── data/                   # Dataset handling
│   ├── datasets.py         #   Kaggle dataset registry
│   ├── loader.py           #   Download & CSV loading
│   └── cleaner.py          #   Cleaning & imputation
├── ui/                     # Streamlit UI components
│   ├── state.py            #   Centralised session state
│   ├── forms.py            #   Dynamic input forms
│   ├── plots.py            #   SHAP/LIME bar charts
│   └── schema.py           #   Schema inference
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
└── requirements.txt
```

## ⚡ Setup & Run

### Prerequisites
- Python 3.12+ (if running locally)
- Docker & Docker Compose (if using containers)
- Kaggle API Credentials (in `.env`)

### Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Running with Docker (Recommended)
1. **Configure credentials**: Create `.env` with your Kaggle username and key.
2. **Launch containers**:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d --build
   ```
3. **Pull the LLM model** (first time only):
   ```bash
   docker exec -it xai-ollama ollama pull llama3.1:8b
   ```
4. **Pull the VLM model** (first time only):
   ```bash
   docker exec -it xai-ollama ollama pull qwen2.5-vl:7b
   ```
5. **Pull the VLM Surrogate model** (first time only):
   ```bash
   docker exec -it xai-ollama ollama pull minicpm-v
   ```
6. **Access the app**: Go to `http://localhost:8501`.

## 🛠️ Memory Optimization

| Model | RAM needed |
|-------|-----------|
| `llama3.1:8b` | ~6.5 GB |
| `qwen2.5-vl:7b` | ~10 GB |
| `minicpm-v` | ~4 GB |



See [DOCKER_GUIDE.md](./DOCKER_GUIDE.md) for WSL2 tuning, GPU setup, and cloud deployment tips.
