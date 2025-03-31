# AI Medical Assistant

A Django-based web application for medical image analysis using deep learning.

## Features
- Multiple disease detection models
- LIME explanations for predictions
- User authentication
- History tracking
- Dark/Light theme

## Setup
1. Clone the repository
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run migrations
```bash
python manage.py migrate
```

5. Start the server
```bash
python manage.py runserver
```

## Models Supported
- Alzheimer's Detection
- Brain Tumor Classification
- Diabetic Retinopathy
- Kidney Disease Detection
- Respiratory Disease Classification 