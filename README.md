# 🔍 AI Imagery Forensic & Deepfake Detector (XAI Dashboard) 🛡️

An Explainable AI (XAI) dashboard for image forensics using the **Faster-Than-Lies (FTL)** methodology. It detects AI-generated (fake) images vs. real camera images by combining traditional RGB representations with forensic feature engineering (FFT Magnitude, LBP Textures, and Sobel Gradients). It compares multiple white-box and black-box visual explanation techniques (Grad-CAM, Saliency, Occlusion, PMI) and leverages a local Large Language Model (LLM) to write forensic analysis reports.

---

## 🌟 Key Features

- **Multi-Page Wizard Workflow**:
  - 🏠 **Home**: Explore forensic principles (FFT, LBP, and Sobel Gradients).
  - ⚙️ **Setup**: Configure model settings, toggling FFT/LBP/Sobel forensic channels, and train a CNN model (on synthetic/Kaggle datasets like CIFAKE).
  - 📤 **Upload**: Upload custom images or select from testing sets to analyze.
  - ⚖️ **XAI Comparison**: Compare heatmaps side-by-side using Grad-CAM, Saliency, Occlusion, and PMI.
  - 📋 **Report**: Generate narrative forensic analysis reports using local LLMs.
- **Forensic Feature Engineering**:
  - **FFT Channel**: Captures periodic frequency anomalies typical of generative models.
  - **LBP Channel**: Detects Local Binary Pattern texture artifacts.
  - **Sobel Channel**: Highlighting edge and gradient magnitude patterns.
- **Dual Explainer Baselines**:
  - **White-box**: Grad-CAM, Input Saliency.
  - **Black-box**: Occlusion, PMI (Pointwise Mutual Information).
- **Narrative Reporting**: Automated Romanian/English bilingual reporting powered by local Ollama instances running `llama3.1:8b-instruct-q4_K_M`.

---

## 📂 Project Structure

```
xai-app/
├── app.py                      # Main entrypoint — Streamlit navigation router
├── pages/                      # Multi-page Streamlit views
│   ├── 1_home.py               #   Overview of forensics & XAI categories
│   ├── 2_setup.py              #   Model settings & training configuration
│   ├── 3_upload.py             #   Image upload & prediction view
│   ├── 4_xai_compare.py        #   Heatmap generation (Grad-CAM, Occlusion, etc.)
│   └── 5_report.py             #   LLM narrative generation
├── core/                       # Core ML, XAI, and LLM implementations
│   ├── fake_data.py            #   Synthetic data generator
│   ├── datasets.py             #   Kaggle dataset integration (CIFAKE)
│   ├── fake_detector.py        #   CNN model, Grad-CAM, Saliency baselines
│   ├── image_features.py       #   LBP, FFT, and Sobel feature extractors
│   ├── image_xai.py            #   Black-box explainer algorithms
│   ├── xai_metrics.py          #   Fidelity & stability evaluation
│   ├── cache.py                #   Model & training state management
│   └── llm.py                  #   Ollama client integration
├── deploy/                     # VM Deployment & automation scripts
│   ├── setup-cloud.sh          #   Azure Ubuntu initialization script
│   ├── deploy.sh               #   Jenkins deployment wrapper script
│   ├── deploy-local.ps1        #   Local PowerShell launch and verify automation
│   └── deploy-cloud.ps1        #   Azure PowerShell update/reset automation
├── docker/                     # Container files
│   ├── Dockerfile.app          #   Streamlit application image
│   ├── Dockerfile.ollama       #   Ollama server image
│   ├── Dockerfile.jenkins      #   CI/CD automation image
│   ├── docker-compose.yml      #   Local orchestration file
│   └── docker-compose.cloud.yml#   Azure VM orchestration file
└── requirements.txt            # Python dependencies
```

---

## ⚡ Setup & Run

### Prerequisites
- Python 3.12+ (if running locally)
- Docker & Docker Compose (recommended)
- Kaggle API Credentials (in `.env` file, only if loading external datasets like CIFAKE)

### Option A: Local Execution (Without Docker)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

### Option B: Containerized Execution (Recommended)
1. **Configure Environment**: Create a `.env` file in the root directory:
   ```env
   KAGGLE_USERNAME=your_username
   KAGGLE_API_TOKEN=your_token
   ```
2. **Launch with PowerShell Automator**:
   ```powershell
   .\deploy\deploy-local.ps1
   ```
   *This automatically starts the docker stack, polls Streamlit until it's ready, and launches it in an Incognito/InPrivate browser window.*

3. **Manual Startup**:
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build
   ```
   Once started, the dashboard is available at: [http://localhost:8501](http://localhost:8501).

---

## ☁️ Azure Cloud Deployment

For VM deployment guidelines, resource bounds, and credentials setup in Jenkins, refer to:
👉 [CLOUD_DEPLOY.md](./CLOUD_DEPLOY.md)
