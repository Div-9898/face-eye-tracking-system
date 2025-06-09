#!/bin/bash

echo "🚀 Starting Face & Eye Tracking System..."
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎯 Launching Streamlit app..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo ""

streamlit run Video_monitoring_app.py 