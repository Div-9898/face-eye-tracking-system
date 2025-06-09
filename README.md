# 👁️ Face & Eye Tracking System

A real-time face and eye tracking application built with Python, Streamlit, and MediaPipe. This system provides advanced facial monitoring capabilities with eye movement detection, blink counting, and comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

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

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera
- Windows/Linux/MacOS

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-eye-tracking-system.git
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

## 🎮 Usage

1. Run the application:
```bash
streamlit run Video_monitoring_app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Grant camera permissions when prompted

4. Click "Start" to begin tracking

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

Your Name - [@yourusername](https://twitter.com/yourusername)

Project Link: [https://github.com/yourusername/face-eye-tracking-system](https://github.com/yourusername/face-eye-tracking-system)

> **Note**: Remember to update the contact information and repository URLs with your actual GitHub username and details!

---

<p align="center">Made with ❤️ using Python and Streamlit</p> 