# Medical ML Project - Render Deployment Guide

This guide provides step-by-step instructions to deploy the Medical ML project on Render.com for free.

---

## Prerequisites

Before deploying, ensure you have:

- [ ] GitHub account
- [ ] Git installed locally
- [ ] Groq API key (get from [console.groq.com](https://console.groq.com/))
- [ ] Project code ready

---

## Part 1: Project Preparation

### 1.1 Verify Required Files

Ensure your project has these files in the root directory:

```
medical-ML project/
├── app/
│   ├── app.py
│   └── templates/
├── src/
├── models/
├── data/
├── requirements.txt    # ✅ Required
├── Procfile            # ✅ Required  
├── runtime.txt         # ✅ Required
└── .env               # ⚠️ Do NOT commit (contains API key)
```

### 1.2 Create .gitignore

Create a `.gitignore` file to exclude sensitive files:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Model files (optional - uncomment if you want to exclude)
# models/*.pth
# models/*.pkl

# OS
.DS_Store
Thumbs.db
```

### 1.3 Test Locally

Before deploying, test the app locally:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
python app/app.py
```

Open http://localhost:5000 and verify:
- [ ] Homepage loads
- [ ] Heart Disease Risk tab works
- [ ] Chest X-Ray tab works
- [ ] Complete Analysis works

---

## Part 2: GitHub Setup

### 2.1 Initialize Git Repository

Open terminal in your project folder:

```bash
# Initialize git
git init

# Add all files
git add .

# Check status
git status
```

### 2.2 Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click **+** → **New repository**
3. Repository name: `medical-ml-project`
4. Description: "AI-powered medical diagnosis system"
5. Set to **Public** or **Private**
6. Click **Create repository**

### 2.3 Push Code to GitHub

```bash
# Add remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/medical-ml-project.git

# First commit
git commit -m "Initial commit - Medical ML project ready for deployment"

# Push to GitHub
git push -u origin main
```

> **Note:** If using `master` branch, replace `main` with `master`

---

## Part 3: Render Deployment

### 3.1 Create Render Account

1. Go to [render.com](https://render.com)
2. Click **Sign Up**
3. Choose **GitHub** to sign in
4. Authorize Render to access your GitHub

### 3.2 Create Web Service

1. In Render dashboard, click **New +**
2. Select **Web Service**
3. Connect your GitHub repository:
   - Find and select `medical-ml-project`
   - Click **Connect**

### 3.3 Configure Build Settings

Fill in the following:

| Setting | Value |
|---------|-------|
| **Name** | `medical-ml` |
| **Region** | Oregon (or closest to you) |
| **Branch** | `main` (or `master`) |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python app/app.py` |

### 3.4 Environment Variables

Scroll to **Environment** section:

1. Click **Add Environment Variable**
2. Add:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | Your Groq API key (from console.groq.com) |

> **Important:** Get your free API key at [console.groq.com](https://console.groq.com/)

### 3.5 Deploy

1. Click **Create Web Service**
2. Wait for deployment (may take 2-5 minutes)
3. View logs in the dashboard

---

## Part 4: Verify Deployment

### 4.1 Check Deployment Status

- Green dot = Success
- Red dot = Failed (check logs)

### 4.2 Test Your Deployed App

Open the URL shown (e.g., `https://medical-ml.onrender.com`)

Test these endpoints:

| Feature | Test |
|---------|------|
| Homepage | Loads without errors |
| Heart Disease | Submit patient data → See "High Risk" or "Low Risk" |
| Chest X-Ray | Upload image → See prediction |
| Complete | Submit both → See predictions + AI report |

### 4.3 Verify API Response

```bash
# Test heart disease prediction
curl -X POST https://YOUR_APP.onrender.com/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{"age":68,"sex":1,"cp":0,"trestbps":160,"chol":234,"fbs":1,"restecg":2,"thalach":131,"exang":0,"oldpeak":0.1,"slope":1,"ca":1,"thal":0}'
```

Expected response:
```json
{
  "prediction": "Low Risk",
  "confidence": 22.0,
  "probabilities": {"No Risk": 78.0, "Risk": 22.0}
}
```

---

## Part 5: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|-----------|
| **Build Failed** | Check requirements.txt has correct packages |
| **Import Error** | Ensure all imports in app.py are available |
| **Model Load Error** | Check models/ folder is in repo |
| **500 Error** | Check GROQ_API_KEY is set correctly |
| **Cold Start Slow** | First request takes 30-50s (normal) |

### View Logs

1. Click on your web service in Render dashboard
2. Go to **Logs** tab
3. Search for error messages

### Redeploy

1. Go to **Deployments** tab
2. Click **New Deployment**
3. Select latest commit

---

## Part 6: Maintenance

### Updating Your App

```bash
# Make changes locally
git add .
git commit -m "Update description"
git push origin main
```

Render auto-deploys on push!

### Scaling

| Plan | Price | Features |
|------|-------|-----------|
| Free | $0 | 750 hrs/month, sleeps after 15 min |
| Starter | $5/mo | No cold starts |
| Pro | $25/mo | More resources |

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your Groq API key |

---

## Security Notes

⚠️ **Important:**
- Never commit `.env` file to GitHub
- Add `.env` to `.gitignore`
- Use Render's environment variables for secrets
- API keys should never be in code

---

## Support

- **Render Docs**: https://render.com/docs
- **Groq API**: https://console.groq.com/docs
- **Project Issues**: Create issue in your GitHub repo

---

## Quick Deploy Checklist

- [ ] Created GitHub repository
- [ ] Pushed code to GitHub
- [ ] Connected repo in Render
- [ ] Set build command: `pip install -r requirements.txt`
- [ ] Set start command: `python app/app.py`
- [ ] Added GROQ_API_KEY environment variable
- [ ] Deployed successfully
- [ ] Tested all features
- [ ] Verified AI report generation works

---

**Deployment Complete!** 🎉

Your Medical ML app is now live at:
```
https://medical-ml.onrender.com
```

*(Replace with your actual Render URL)*

---

*Last updated: April 2026*