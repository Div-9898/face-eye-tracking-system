# 👁️ Face & Eye Tracking System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-eye-tracking-system.streamlit.app)
[![GitHub](https://img.shields.io/github/stars/Div-9898/face-eye-tracking-system?style=social)](https://github.com/Div-9898/face-eye-tracking-system)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)

> **🚀 [Try the Live App](https://face-eye-tracking-system.streamlit.app)** | Real-time face and eye tracking in your browser!

A real-time face and eye tracking application built with Python, Streamlit, and MediaPipe. This system provides advanced facial monitoring capabilities with eye movement detection, blink counting, and comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Quick Start

### 🌐 Run Online (No Installation)
Click here to run the app directly in your browser:
- **[Launch App on Streamlit Cloud →](https://face-eye-tracking-system.streamlit.app)**

### 💻 Run Locally
```bash
git clone https://github.com/Div-9898/face-eye-tracking-system.git
cd face-eye-tracking-system
pip install -r requirements.txt
streamlit run Video_monitoring_app.py
```

## ✨ Features

- **Real-time Face Detection**: Accurate face tracking using MediaPipe's advanced AI models
- **Eye Movement Tracking**: Detects both eyes individually with Eye Aspect Ratio (EAR) calculations
- **Blink Detection**: Counts blinks and monitors eye closure patterns
- **Head Pose Estimation**: Tracks pitch, yaw, and roll angles to determine gaze direction
- **Center Position Tracking**: Monitors if the face is centered in the camera frame
- **Live Analytics**: Real-time performance metrics and visualizations
- **Session Recording**: Save and export tracking data for later analysis
- **Beautiful UI**: Modern, animated interface with gradient themes

## 🖼️ Screenshots

### Main Dashboard
- Real-time camera feed with face landmarks
- Live status indicators
- Performance analytics
- Eye tracking visualizations

### Features Include:
- 📊 Circular progress indicators for metrics
- 📈 Real-time charts for eye aspect ratios
- 🎯 Distance from center tracking
- 👀 Individual eye detection status
- 🔄 Head orientation gauges

## 🚀 Run the App

### 🌐 Method 1: Streamlit Cloud (Easiest - No Installation!)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-eye-tracking-system.streamlit.app)

Click the badge above or visit: https://face-eye-tracking-system.streamlit.app

**Note**: Camera features work best locally due to browser security restrictions.

### 💻 Method 2: GitHub Codespaces (Cloud Development)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Div-9898/face-eye-tracking-system)

1. Click the badge above
2. Wait for environment setup
3. Run: `streamlit run Video_monitoring_app.py`
4. The app will open automatically

### 🐙 Method 3: Gitpod (Alternative Cloud IDE)
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/Div-9898/face-eye-tracking-system)

### 🖥️ Method 4: Local Installation

#### Prerequisites
- Python 3.8 or higher
- Webcam/Camera
- Windows/Linux/MacOS

#### Quick Setup
```bash
# Clone and enter directory
git clone https://github.com/Div-9898/face-eye-tracking-system.git
cd face-eye-tracking-system

# Run the setup script (Linux/Mac)
chmod +x run.sh
./run.sh

# Or manually:
pip install -r requirements.txt
streamlit run Video_monitoring_app.py
```

#### Detailed Setup
1. Clone the repository:
```bash
git clone https://github.com/Div-9898/face-eye-tracking-system.git
cd face-eye-tracking-system
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run Video_monitoring_app.py
```

## 🎮 Usage

### Local Usage

1. Run the application:
```bash
streamlit run Video_monitoring_app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Grant camera permissions when prompted

4. Click "Start" to begin tracking

### 🌐 Deploy on Streamlit Cloud

You can also deploy this app on Streamlit Cloud for free:

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your forked repository
4. Choose `streamlit_app.py` as the main file
5. **⚠️ CRITICAL**: Click "Advanced settings" and select **Python 3.11** from the dropdown
   - MediaPipe does NOT support Python 3.13+ yet
   - You MUST select Python 3.11 or the deployment will fail
6. Click "Deploy"

**🔧 If you already deployed and it's failing:**
1. Go to your app dashboard on Streamlit Cloud
2. Click the three dots menu (⋮) → Settings
3. Under "General" → "Python version", select **Python 3.11**
4. Click "Save" and the app will redeploy automatically

**Note**: Camera access in Streamlit Cloud depends on browser permissions and HTTPS. The app works best when run locally or on HTTPS-enabled domains.

### 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Video_monitoring_app.py"]
```

### Controls
- **▶️ Start**: Begin face tracking session
- **⏸️ Stop**: Pause tracking and save session data
- **📸 Snapshot**: Capture current frame (coming soon)
- **🔄 Reset**: Clear counters and data
- **🚪 Exit**: Close the application

### Settings
- **Center Threshold**: Adjust sensitivity for center detection
- **Blink Threshold**: Set Eye Aspect Ratio threshold for blink detection
- **Pitch/Yaw Thresholds**: Configure head pose detection sensitivity
- **Debug Mode**: Show/hide FPS and detection information
- **Show Face Landmarks**: Toggle face mesh visualization

## 📊 Metrics Explained

### Eye Aspect Ratio (EAR)
- Measures eye openness
- Values typically range from 0.1 to 0.4
- Lower values indicate closed eyes

### Head Pose Angles
- **Pitch**: Up/down head movement
- **Yaw**: Left/right head rotation
- **Roll**: Head tilt angle

### Performance Metrics
- **Face Detection Rate**: Percentage of frames with detected face
- **Centered Rate**: Time spent in center position
- **Attention Rate**: Time spent looking at camera
- **Both Eyes Rate**: Frames with both eyes detected

## 🛠️ Technical Details

### Technologies Used
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face detection and landmark tracking
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations

### Key Components
1. **FaceTracker Class**: Core tracking logic and calculations
2. **StreamlitApp Class**: UI management and user interactions
3. **Real-time Processing**: Frame-by-frame analysis at 30 FPS
4. **Session Analytics**: Data collection and export functionality

## 📁 Project Structure

```
face-eye-tracking-system/
│
├── Video_monitoring_app.py    # Main application file
├── test_camera.py            # Camera testing utility
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── .gitignore              # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for the amazing face detection models
- Streamlit for the excellent web framework
- OpenCV community for computer vision tools

## 📞 Contact

GitHub: [@Div-9898](https://github.com/Div-9898)

Project Link: [https://github.com/Div-9898/face-eye-tracking-system](https://github.com/Div-9898/face-eye-tracking-system)

---

<p align="center">Made with ❤️ using Python and Streamlit</p> 
