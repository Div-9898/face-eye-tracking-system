import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import threading
import queue
from PIL import Image
import io

# Page config is already set in streamlit_app.py, so we don't set it here
# Last updated: 2025-01-09 - Fixed set_page_config error

# Apply global CSS theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Container Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    div[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric Container Styles */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Success/Error/Warning Styles */
    .element-container div[data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Slider Styles */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        backdrop-filter: blur(5px);
        color: white !important;
    }
    
    /* Container Animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Glow Effect for Active Elements */
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
        100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
    }
    
    .tracking-active {
        animation: glow 2s infinite;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Info Boxes */
    .stInfo {
        background: rgba(52, 152, 219, 0.1) !important;
        color: white !important;
        border-left: 4px solid #3498db;
    }
    
    .stSuccess {
        background: rgba(46, 204, 113, 0.1) !important;
        color: white !important;
        border-left: 4px solid #2ecc71;
    }
    
    .stWarning {
        background: rgba(241, 196, 15, 0.1) !important;
        color: white !important;
        border-left: 4px solid #f1c40f;
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.1) !important;
        color: white !important;
        border-left: 4px solid #e74c3c;
    }
    
    /* Pulse Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

class FaceTracker:
    """Main class for face detection and tracking"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh model with better parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Lowered for better detection
            min_tracking_confidence=0.5     # Lowered for better tracking
        )
        
        # Improved eye landmarks indices for better eye detection
        # Using MediaPipe's official eye contour indices
        
        # Right eye indices (from viewer's perspective, actually person's left eye)
        self.RIGHT_EYE_INDICES = [
            # Eye contour
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            173, 157, 158, 159, 160, 161, 246
        ]
        
        # Left eye indices (from viewer's perspective, actually person's right eye)
        self.LEFT_EYE_INDICES = [
            # Eye contour
            362, 398, 384, 385, 386, 387, 388, 466,
            263, 249, 390, 373, 374, 380, 381, 382
        ]
        
        # Key points for EAR calculation (using 6 points)
        self.RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  # P1, P2, P3, P4, P5, P6
        self.LEFT_EYE_POINTS = [362, 385, 387, 263, 380, 374]  # P1, P2, P3, P4, P5, P6
        
        # Iris landmarks for better eye tracking
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Face oval landmarks
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Center position tracking
        self.center_threshold = 50  # pixels
        
        # Head pose thresholds
        self.pitch_threshold = 30  # degrees
        self.yaw_threshold = 30    # degrees
        
        # Eye tracking parameters
        self.EAR_THRESHOLD = 0.21  # Eye aspect ratio threshold for blink detection
        self.CONSECUTIVE_FRAMES = 2  # Consecutive frames for blink
        self.blink_counter = 0
        self.total_blinks = 0
        self.eye_closed_frames = 0
        
        # Data storage
        self.tracking_data = deque(maxlen=300)  # 10 seconds at 30fps
        
        # Debug mode
        self.debug_mode = True
        
    def validate_landmark_index(self, index, total_landmarks):
        """Validate if a landmark index is within bounds"""
        return 0 <= index < total_landmarks
    
    def get_eye_landmarks(self, landmarks, eye_indices, img_w, img_h):
        """Extract eye landmarks as numpy array with validation"""
        eye_landmarks = []
        total_landmarks = len(landmarks)
        
        for idx in eye_indices:
            if self.validate_landmark_index(idx, total_landmarks):
                try:
                    x = int(landmarks[idx].x * img_w)
                    y = int(landmarks[idx].y * img_h)
                    eye_landmarks.append([x, y])
                except Exception as e:
                    print(f"Error extracting landmark {idx}: {e}")
                    continue
            else:
                print(f"Invalid landmark index: {idx} (total landmarks: {total_landmarks})")
        
        return np.array(eye_landmarks) if eye_landmarks else np.array([])
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        if len(eye_landmarks) < 6:
            return 0
        
        try:
            # Compute the euclidean distances between the two sets of vertical eye landmarks
            # Using the improved landmark positions
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Compute the euclidean distance between the horizontal eye landmarks
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Avoid division by zero
            if C == 0:
                return 0
            
            # Compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0
    
    def draw_eye_region(self, frame, eye_landmarks, color=(0, 255, 0), label="", show_indices=False, indices=None):
        """Draw eye region with landmarks and bounding box"""
        if len(eye_landmarks) < 4:
            return
        
        # Draw eye contour
        eye_hull = cv2.convexHull(eye_landmarks)
        cv2.drawContours(frame, [eye_hull], -1, color, 1)
        
        # Draw individual landmarks
        for i, point in enumerate(eye_landmarks):
            cv2.circle(frame, tuple(point), 2, color, -1)
            
            # Show indices in calibration mode
            if show_indices and indices and i < len(indices):
                cv2.putText(frame, str(indices[i]), 
                           (point[0] + 5, point[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(eye_landmarks)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        
        # Add label if provided
        if label:
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def detect_blink(self, left_ear, right_ear):
        """Detect if a blink occurred based on EAR values"""
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < self.EAR_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.CONSECUTIVE_FRAMES:
                self.total_blinks += 1
            self.eye_closed_frames = 0
        
        return avg_ear < self.EAR_THRESHOLD
    
    def get_head_pose(self, landmarks, img_w, img_h):
        """Calculate head pose using facial landmarks"""
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),      # Nose tip
            (landmarks[175].x * img_w, landmarks[175].y * img_h),  # Chin
            (landmarks[33].x * img_w, landmarks[33].y * img_h),    # Left eye left corner
            (landmarks[263].x * img_w, landmarks[263].y * img_h),  # Right eye right corner
            (landmarks[61].x * img_w, landmarks[61].y * img_h),    # Left mouth corner
            (landmarks[291].x * img_w, landmarks[291].y * img_h)   # Right mouth corner
        ], dtype="double")
        
        # Camera internals
        focal_length = img_w
        center = (img_w/2, img_h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = math.atan2(-rotation_matrix[2,0], sy)
            z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = math.atan2(-rotation_matrix[2,0], sy)
            z = 0
        
        return np.degrees([x, y, z])
    
    def is_looking_at_camera(self, landmarks, img_w, img_h):
        """Determine if user is looking at camera with improved detection"""
        try:
            # Get head pose
            angles = self.get_head_pose(landmarks, img_w, img_h)
            
            # Check if head is facing forward
            pitch, yaw, roll = angles
            
            # Use class thresholds
            base_pitch_threshold = self.pitch_threshold
            base_yaw_threshold = self.yaw_threshold
            
            # Get nose tip position relative to face center
            nose_tip = landmarks[1]
            face_center_x = np.mean([landmark.x for landmark in landmarks])
            face_center_y = np.mean([landmark.y for landmark in landmarks])
            
            # Calculate nose offset from face center (indicates gaze direction)
            nose_offset_x = abs(nose_tip.x - face_center_x)
            nose_offset_y = abs(nose_tip.y - face_center_y)
            
            # More lenient if nose is centered (good indicator of looking at camera)
            if nose_offset_x < 0.05 and nose_offset_y < 0.05:
                pitch_threshold = base_pitch_threshold * 1.5
                yaw_threshold = base_yaw_threshold * 1.5
            else:
                pitch_threshold = base_pitch_threshold
                yaw_threshold = base_yaw_threshold
            
            # Consider looking at camera based on head pose
            looking_at_camera = (abs(pitch) < pitch_threshold and abs(yaw) < yaw_threshold)
            
            return looking_at_camera, angles
        except Exception as e:
            print(f"Error in is_looking_at_camera: {e}")
            return False, [0, 0, 0]
    
    def is_centered(self, landmarks, img_w, img_h):
        """Check if face is centered in frame"""
        # Get face center
        face_center_x = np.mean([landmark.x for landmark in landmarks]) * img_w
        face_center_y = np.mean([landmark.y for landmark in landmarks]) * img_h
        
        # Image center
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        # Calculate distance from center
        distance = np.sqrt((face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2)
        
        is_centered = distance < self.center_threshold
        
        return is_centered, (face_center_x, face_center_y), distance
    
    def process_frame(self, frame):
        """Process a single frame and return tracking data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        img_h, img_w = frame.shape[:2]
        
        tracking_info = {
            'timestamp': time.time(),
            'face_detected': False,
            'is_centered': False,
            'looking_at_camera': False,
            'center_distance': 0,
            'head_angles': [0, 0, 0],
            'face_center': (0, 0),
            'left_eye_detected': False,
            'right_eye_detected': False,
            'left_ear': 0,
            'right_ear': 0,
            'is_blinking': False,
            'total_blinks': self.total_blinks
        }
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Update tracking info
                tracking_info['face_detected'] = True
                
                # Check if centered
                is_centered, face_center, distance = self.is_centered(landmarks, img_w, img_h)
                tracking_info['is_centered'] = is_centered
                tracking_info['center_distance'] = distance
                tracking_info['face_center'] = face_center
                
                # Check if looking at camera
                looking_at_camera, angles = self.is_looking_at_camera(landmarks, img_w, img_h)
                tracking_info['looking_at_camera'] = looking_at_camera
                tracking_info['head_angles'] = angles
                
                # Extract eye landmarks
                left_eye_landmarks = self.get_eye_landmarks(landmarks, self.LEFT_EYE_INDICES, img_w, img_h)
                right_eye_landmarks = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES, img_w, img_h)
                
                # Get specific points for EAR calculation
                left_ear_points = self.get_eye_landmarks(landmarks, self.LEFT_EYE_POINTS, img_w, img_h)
                right_ear_points = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_POINTS, img_w, img_h)
                
                # Calculate EAR for both eyes
                if len(left_ear_points) >= 6:
                    left_ear = self.calculate_eye_aspect_ratio(left_ear_points)
                    tracking_info['left_eye_detected'] = True
                    tracking_info['left_ear'] = left_ear
                else:
                    left_ear = 0
                
                if len(right_ear_points) >= 6:
                    right_ear = self.calculate_eye_aspect_ratio(right_ear_points)
                    tracking_info['right_eye_detected'] = True
                    tracking_info['right_ear'] = right_ear
                else:
                    right_ear = 0
                
                # Detect blink
                is_blinking = self.detect_blink(left_ear, right_ear)
                tracking_info['is_blinking'] = is_blinking
                
                # Draw face mesh contours (lighter)
                if hasattr(st.session_state, 'show_landmarks') and st.session_state.show_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
                    )
                
                # Draw eye regions with enhanced visualization
                if tracking_info['left_eye_detected']:
                    color = (0, 0, 255) if left_ear < self.EAR_THRESHOLD else (0, 255, 0)
                    self.draw_eye_region(frame, left_eye_landmarks, color, f"Left EAR: {left_ear:.2f}")
                
                if tracking_info['right_eye_detected']:
                    color = (0, 0, 255) if right_ear < self.EAR_THRESHOLD else (0, 255, 0)
                    self.draw_eye_region(frame, right_eye_landmarks, color, f"Right EAR: {right_ear:.2f}")
                
                # Draw face center point
                cv2.circle(frame, (int(face_center[0]), int(face_center[1])), 5, (0, 255, 0), -1)
                
                # Draw frame center and threshold circle
                cv2.circle(frame, (img_w//2, img_h//2), 5, (255, 0, 0), -1)
                cv2.circle(frame, (img_w//2, img_h//2), self.center_threshold, (255, 0, 0), 2)
                
                # Add debug information if enabled
                if self.debug_mode:
                    # Calculate actual FPS
                    current_time = time.time()
                    if hasattr(self, 'last_time'):
                        fps = 1 / (current_time - self.last_time)
                    else:
                        fps = 0
                    self.last_time = current_time
                    
                    debug_text = [
                        f"FPS: {fps:.1f}",
                        f"Blinks: {self.total_blinks}",
                        f"Left Eye: {'Detected' if tracking_info['left_eye_detected'] else 'Not Detected'}",
                        f"Right Eye: {'Detected' if tracking_info['right_eye_detected'] else 'Not Detected'}"
                    ]
                    
                    for i, text in enumerate(debug_text):
                        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Store tracking data
        self.tracking_data.append(tracking_info)
        
        return frame, tracking_info

class StreamlitApp:
    """Main Streamlit application"""
    
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.camera_active = False
        self.frame_queue = queue.Queue(maxsize=10)
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'camera_permission' not in st.session_state:
            st.session_state.camera_permission = False
        if 'tracking_active' not in st.session_state:
            st.session_state.tracking_active = False
        if 'session_analytics' not in st.session_state:
            st.session_state.session_analytics = []
        if 'show_landmarks' not in st.session_state:
            st.session_state.show_landmarks = True
        if 'tracking_history' not in st.session_state:
            st.session_state.tracking_history = []
        if 'show_permission_popup' not in st.session_state:
            st.session_state.show_permission_popup = True
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
    
    def request_camera_permission(self):
        """Request camera access from user with enhanced UI"""
        # Add loading animation and styles
        st.markdown("""
        <style>
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
                100% { transform: translateY(0px); }
            }
            
            .floating {
                animation: float 3s ease-in-out infinite;
            }
            
            .hero-container {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 30px;
                padding: 3rem;
                margin: 2rem auto;
                max-width: 800px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                text-align: center;
            }
            
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .feature-card {
                background: rgba(255, 255, 255, 0.05);
                padding: 1.5rem;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                background: rgba(255, 255, 255, 0.1);
            }
            
            .emoji-icon {
                font-size: 3rem;
                display: block;
                margin-bottom: 1rem;
            }
            
            .permission-popup {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 2rem;
                margin: 2rem auto;
                max-width: 500px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                border: 2px solid rgba(102, 126, 234, 0.3);
                text-align: center;
                animation: fadeIn 0.5s ease-out;
            }
            
            @keyframes pulse-border {
                0% { border-color: rgba(102, 126, 234, 0.3); }
                50% { border-color: rgba(102, 126, 234, 0.8); }
                100% { border-color: rgba(102, 126, 234, 0.3); }
            }
            
            .permission-popup:hover {
                animation: pulse-border 2s infinite;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Hero section
        st.markdown("""
        <div class="hero-container">
            <div class="floating">
                <span style="font-size: 5rem;">👁️</span>
            </div>
            <h1 style="font-size: 2.5rem; margin: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Face & Eye Tracking System
            </h1>
            <p style="font-size: 1.2rem; color: #888; margin-bottom: 2rem;">
                Advanced AI-powered facial monitoring with real-time analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span class="emoji-icon">✨</span>
                <h3>Real-time Detection</h3>
                <p>Instant face and eye tracking with advanced AI</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <span class="emoji-icon">📊</span>
                <h3>Live Analytics</h3>
                <p>Performance metrics and session insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <span class="emoji-icon">👀</span>
                <h3>Blink Detection</h3>
                <p>Monitor eye movements and blink patterns</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <span class="emoji-icon">📈</span>
                <h3>Data Export</h3>
                <p>Save and analyze your tracking sessions</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Camera permission popup section
        if 'show_permission_popup' not in st.session_state:
            st.session_state.show_permission_popup = True
        
        if st.session_state.show_permission_popup:
            # Show permission popup
            st.markdown("""
            <div class="permission-popup">
                <span style="font-size: 4rem;">📷</span>
                <h2 style="color: #333; margin: 1rem 0;">Camera Access Setup</h2>
                <p style="color: #666; margin-bottom: 1.5rem;">
                    To track your face and eyes, we need camera permission.
                    Your privacy is protected - video is processed locally only.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Instructions section
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.9); 
                       border-radius: 15px; 
                       padding: 1.5rem; 
                       margin: 1rem auto;
                       max-width: 600px;
                       border: 2px solid rgba(102, 126, 234, 0.3);">
                <h3 style="color: #333; margin-top: 0;">📋 Quick Setup Instructions:</h3>
                <ol style="color: #555; text-align: left;">
                    <li>Click the <strong>camera button</strong> below</li>
                    <li>Your browser will ask for camera permission</li>
                    <li>Click <strong>"Allow"</strong> in the popup</li>
                    <li>Take a quick photo or just click the button</li>
                    <li>You'll be redirected to the main dashboard</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                # Use camera_input which automatically triggers browser permission dialog
                st.markdown("### 👇 Click the Camera Button Below")
                
                camera_test = st.camera_input(
                    "Click to enable camera and grant permission", 
                    key="auto_camera_permission",
                    help="This will open your camera and request permission"
                )
                
                if camera_test is not None:
                    # Camera permission granted!
                    st.session_state.camera_permission = True
                    st.session_state.show_permission_popup = False
                    st.balloons()
                    st.success("🎉 Perfect! Camera access granted!")
                    st.info("🚀 Loading the Face Tracking Dashboard...")
                    time.sleep(1.5)
                    st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Alternative options
                with st.expander("⚡ Quick Access Options"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("🚀 I've already granted permission", use_container_width=True):
                            # Test if camera is actually accessible
                            try:
                                cap = cv2.VideoCapture(0)
                                if cap.isOpened():
                                    # Test reading a frame to ensure it's working
                                    ret, frame = cap.read()
                                    cap.release()
                                    if ret:
                                        st.session_state.camera_permission = True
                                        st.session_state.show_permission_popup = False
                                        st.session_state.camera_tested = True  # Mark as tested
                                        st.success("✅ Camera detected! Loading...")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error("Camera opened but couldn't read frames. Please use the camera button above.")
                                else:
                                    st.error("Camera not accessible. Please use the camera button above.")
                            except Exception as e:
                                st.error(f"Camera error: {str(e)}. Please use the camera button above.")
                    
                    with col_b:
                        if st.button("💻 Continue without camera", use_container_width=True):
                            st.warning("⚠️ Camera is required for face tracking features")
                            if st.checkbox("I understand the app won't work properly"):
                                st.session_state.camera_permission = True
                                st.session_state.show_permission_popup = False
                                st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("📋 View Privacy Policy", use_container_width=True):
                    with st.expander("Privacy Policy", expanded=True):
                        st.markdown("""
                        ### 🔒 Your Privacy Matters
                        
                        - **No Recording**: Video is processed in real-time only
                        - **No Storage**: No images or videos are saved without explicit consent
                        - **Local Processing**: All analysis happens on your device
                        - **Data Control**: You can export or delete your data anytime
                        - **Open Source**: Full transparency with public code
                        """)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Manual permission instructions
                with st.expander("🔧 Manual Permission Setup"):
                    st.markdown("""
                    If the automatic permission doesn't work:
                    
                    **Chrome/Edge:**
                    1. Click the camera icon in the address bar
                    2. Select "Allow" for camera access
                    3. Refresh the page
                    
                    **Firefox:**
                    1. Click the lock icon in the address bar
                    2. Click ">" next to "Permissions"
                    3. Allow camera access
                    4. Refresh the page
                    
                    **Safari:**
                    1. Go to Safari → Preferences → Websites → Camera
                    2. Find this website and select "Allow"
                    3. Refresh the page
                    """)
        else:
            # Permission already handled, show loading
            st.info("🚀 Camera permission granted. Loading application...")
    
    def _is_running_locally(self):
        """Check if the app is running locally or in the cloud"""
        # Check for common cloud indicators
        is_cloud = any([
            os.getenv('STREAMLIT_CLOUD_VENDOR') is not None,
            os.getenv('STREAMLIT_SHARING_MODE') is not None,
            os.getenv('IS_DOCKER_CONTAINER') is not None,
            # Check if we can access local camera (this will fail in cloud)
            'localhost' not in str(st.get_option('server.address')) if hasattr(st, 'get_option') else False
        ])
        
        # Additional check - try to open camera quickly
        if not is_cloud:
            try:
                test_cap = cv2.VideoCapture(0)
                if test_cap.isOpened():
                    test_cap.release()
                    return True
                else:
                    return False
            except:
                return False
        
        return not is_cloud
    
    def _run_browser_tracking(self, camera_placeholder, status_placeholder, analytics_placeholder):
        """Run tracking in browser mode using frame-by-frame capture"""
        with camera_placeholder.container():
            st.warning("🌐 Running in browser mode - Real-time video tracking is limited")
            st.info("📸 Use the camera button below to capture frames for analysis")
            
            # Create columns for controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Camera input for frame capture
                captured_image = st.camera_input(
                    "Click to capture frame for analysis",
                    key=f"browser_camera_{st.session_state.frame_count}"
                )
            
            with col2:
                st.metric("Frames Analyzed", st.session_state.frame_count)
            
            with col3:
                if st.button("🔄 Reset", key="reset_frames"):
                    st.session_state.frame_count = 0
                    self.face_tracker.total_blinks = 0
                    self.face_tracker.tracking_data.clear()
                    st.rerun()
            
            if captured_image is not None:
                # Process the captured image
                try:
                    # Convert uploaded image to OpenCV format
                    image = Image.open(captured_image)
                    image_array = np.array(image)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(image_array.shape) == 3:
                        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        frame = image_array
                    
                    # Process frame
                    processed_frame, tracking_info = self.face_tracker.process_frame(frame)
                    
                    # Display processed frame
                    st.image(processed_frame, channels="BGR", use_column_width=True)
                    
                    # Update status and analytics
                    self.update_status(status_placeholder, tracking_info)
                    self.update_analytics(analytics_placeholder)
                    
                    # Increment frame count
                    st.session_state.frame_count += 1
                    
                    # Auto-refresh to allow continuous capture
                    st.info("✅ Frame processed! Take another photo to continue tracking.")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            
            # Show instructions
            with st.expander("📖 Browser Mode Instructions", expanded=False):
                st.markdown("""
                ### How to use Browser Mode:
                
                1. **Click the camera button** to capture a frame
                2. **Allow camera access** when prompted by your browser
                3. **Take a photo** - the frame will be analyzed automatically
                4. **Repeat** to capture more frames for continuous tracking
                
                ### Limitations:
                - No real-time video feed (frame-by-frame only)
                - Analytics update after each captured frame
                - Blink detection may be less accurate
                
                ### For Real-time Tracking:
                - Download and run the app locally
                - Use the command: `streamlit run Video_monitoring_app.py`
                """)
    
    def create_dashboard(self):
        """Main dashboard interface"""
        # Header with gradient
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3rem; margin: 0; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                👁️ Face & Eye Tracking Dashboard
            </h1>
            <p style="color: #888; font-size: 1.2rem; margin-top: 0.5rem;">
                Real-time monitoring and analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show running mode
        is_local = self._is_running_locally()
        if not is_local:
            st.info("🌐 Running in **Browser Mode** - Frame-by-frame analysis only. For real-time video tracking, run the app locally.")
        else:
            st.success("💻 Running in **Local Mode** - Full real-time video tracking available!")
        
        # Show session analytics if available and not tracking
        if not st.session_state.tracking_active and st.session_state.session_analytics:
            with st.container():
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.05); 
                           backdrop-filter: blur(10px); 
                           border-radius: 20px; 
                           padding: 2rem; 
                           margin-bottom: 2rem;
                           border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h3 style="color: white; margin-top: 0;">📊 Last Session Summary</h3>
                </div>
                """, unsafe_allow_html=True)
                self.show_session_summary()
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Sidebar settings with modern design
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: white; margin: 0;">⚙️ Control Panel</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Detection settings
            with st.expander("🎯 Detection Settings", expanded=True):
                self.face_tracker.center_threshold = st.slider(
                    "Center Threshold (pixels)", 20, 100, 50,
                    help="Distance from center to be considered 'centered'"
                )
                self.face_tracker.EAR_THRESHOLD = st.slider(
                    "Blink Threshold (EAR)", 0.1, 0.3, 0.21, 0.01,
                    help="Eye Aspect Ratio threshold for blink detection"
                )
                
                # Head pose thresholds
                st.markdown("**🔄 Head Pose Thresholds**")
                self.face_tracker.pitch_threshold = st.slider("Pitch Threshold", 10, 50, 30,
                    help="Maximum pitch angle to be considered 'looking at camera'")
                self.face_tracker.yaw_threshold = st.slider("Yaw Threshold", 10, 50, 30,
                    help="Maximum yaw angle to be considered 'looking at camera'")
            
            # Display settings
            with st.expander("🎨 Display Options", expanded=True):
                self.face_tracker.debug_mode = st.checkbox("Show Debug Info", True)
                st.session_state.show_landmarks = st.checkbox("Show Face Landmarks", True)
                
                st.info("💡 Debug info shows FPS, blink count, and eye detection status")
            
            # Camera settings
            with st.expander("📷 Camera Settings", expanded=False):
                require_test = st.checkbox("Require camera test before tracking", 
                                         value=st.session_state.get('require_camera_test', True),
                                         key="require_camera_test",
                                         help="When enabled, you'll need to take a test photo before tracking starts")
                
                if st.button("🔄 Reset Camera Permission", use_container_width=True):
                    if 'camera_tested' in st.session_state:
                        del st.session_state.camera_tested
                    st.success("Camera permission reset. You'll be asked to test camera next time.")
                
                st.info("💡 Disable camera test if you're having issues with repeated permission prompts")
            
            # Data export
            with st.expander("💾 Data Export", expanded=False):
                if st.button("📥 Export Current Data", use_container_width=True):
                    self.export_tracking_data()
                
                if st.button("🗑️ Clear Session History", use_container_width=True):
                    st.session_state.session_analytics = []
                    st.session_state.tracking_history = []
                    st.success("✅ History cleared!")
            
            # Session info
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <p style="color: #888; margin: 0;">Session Status</p>
                <h3 style="color: """ + ("#2ecc71" if st.session_state.tracking_active else "#e74c3c") + """; margin: 0.5rem 0;">
                    """ + ("🟢 Active" if st.session_state.tracking_active else "🔴 Inactive") + """
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content area with modern layout
        main_container = st.container()
        
        with main_container:
            # Control buttons with modern style
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); 
                       backdrop-filter: blur(10px); 
                       border-radius: 20px; 
                       padding: 1.5rem; 
                       margin-bottom: 2rem;
                       border: 1px solid rgba(255, 255, 255, 0.1);">
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            
            with col1:
                if st.button("▶️ Start", type="primary", disabled=st.session_state.tracking_active, 
                           use_container_width=True, key="start_btn"):
                    st.session_state.tracking_active = True
                    st.session_state.tracking_start_time = time.time()
                    st.rerun()
            
            with col2:
                if st.button("⏸️ Stop", disabled=not st.session_state.tracking_active, 
                           use_container_width=True, key="stop_btn"):
                    self.save_session_analytics()
                    st.session_state.tracking_active = False
                    # Reset camera test flag so it can be tested again next time
                    if 'camera_tested' in st.session_state:
                        del st.session_state.camera_tested
                    # Reset frame count for browser mode
                    st.session_state.frame_count = 0
                    st.rerun()
            
            with col3:
                if st.button("📸 Snapshot", disabled=not st.session_state.tracking_active, 
                           use_container_width=True):
                    st.toast("📸 Snapshot feature coming soon!", icon="📸")
            
            with col4:
                if st.button("🔄 Reset", use_container_width=True):
                    self.face_tracker.total_blinks = 0
                    self.face_tracker.tracking_data.clear()
                    st.toast("✅ Counters reset!", icon="✅")
            
            with col5:
                if st.button("🚪 Exit", use_container_width=True):
                    st.session_state.camera_permission = False
                    st.session_state.tracking_active = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Main content columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Camera feed section with modern card design
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.05); 
                           backdrop-filter: blur(10px); 
                           border-radius: 20px; 
                           padding: 1.5rem;
                           border: 1px solid rgba(255, 255, 255, 0.1);
                           """ + ("box-shadow: 0 0 30px rgba(102, 126, 234, 0.5);" if st.session_state.tracking_active else "") + """">
                    <h3 style="color: white; margin-top: 0;">📹 Live Camera Feed</h3>
                </div>
                """, unsafe_allow_html=True)
                
                camera_placeholder = st.empty()
            
            with col2:
                # Status section with modern card design
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.05); 
                           backdrop-filter: blur(10px); 
                           border-radius: 20px; 
                           padding: 1.5rem;
                           margin-bottom: 1rem;
                           border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h3 style="color: white; margin-top: 0;">📊 Real-time Status</h3>
                </div>
                """, unsafe_allow_html=True)
                
                status_placeholder = st.empty()
                
                # Analytics section
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.05); 
                           backdrop-filter: blur(10px); 
                           border-radius: 20px; 
                           padding: 1.5rem;
                           margin-top: 1rem;
                           border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h3 style="color: white; margin-top: 0;">📈 Live Analytics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                analytics_placeholder = st.empty()
        
        # Run tracking if active
        if st.session_state.tracking_active:
            self.run_tracking_loop(camera_placeholder, status_placeholder, analytics_placeholder)
        else:
            # Show placeholders with animations
            with camera_placeholder.container():
                st.markdown("""
                <div style="height: 400px; 
                           display: flex; 
                           align-items: center; 
                           justify-content: center;
                           background: rgba(255, 255, 255, 0.02);
                           border-radius: 15px;
                           border: 2px dashed rgba(255, 255, 255, 0.2);">
                    <div style="text-align: center;">
                        <div class="floating">
                            <span style="font-size: 4rem; opacity: 0.5;">📷</span>
                        </div>
                        <p style="color: #888; margin-top: 1rem;">Camera feed will appear here</p>
                        <p style="color: #666;">Click 'Start' to begin tracking</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with status_placeholder.container():
                st.info("📊 Status information will appear here during tracking")
            
            with analytics_placeholder.container():
                st.info("📈 Analytics will be displayed here during tracking")
    
    def run_tracking_loop(self, camera_placeholder, status_placeholder, analytics_placeholder):
        """Main tracking loop"""
        # Check if we're running locally or in the cloud
        is_local = self._is_running_locally()
        
        if not is_local:
            # Cloud/browser deployment - use frame-by-frame mode
            self._run_browser_tracking(camera_placeholder, status_placeholder, analytics_placeholder)
            return
        
        # Local deployment - use OpenCV for real-time tracking
        # First, try to test camera access with Streamlit's camera_input if required
        require_test = st.session_state.get('require_camera_test', True)
        
        if require_test and 'camera_tested' not in st.session_state:
            with camera_placeholder.container():
                st.info("🎥 Initializing camera access...")
                
                # Check if we should skip the camera test
                skip_test = st.checkbox("Skip camera test (I know my camera is working)", key="skip_camera_test")
                
                if skip_test:
                    st.session_state.camera_tested = True
                    st.rerun()
                else:
                    # Use camera_input to ensure browser permissions
                    st.markdown("### 📸 Quick Camera Test")
                    st.info("Take a quick photo to confirm camera access. This ensures your browser has granted camera permissions.")
                    
                    test_image = st.camera_input("Click to test camera", key="dashboard_camera_test")
                    if test_image is not None:
                        st.session_state.camera_tested = True
                        st.success("✅ Camera access confirmed! Starting tracking...")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning("⚠️ Waiting for camera test...")
                        
                        # Provide alternative option
                        if st.button("🚀 Skip and try anyway", key="skip_test_button"):
                            st.session_state.camera_tested = True
                            st.rerun()
                    return
        
        # Now try to open camera with OpenCV
        cap = None
        camera_opened = False
        
        # Try different methods to open camera
        for method in ['default', 'dshow', 'msmf']:
            for camera_index in [0, 1, 2]:
                try:
                    if method == 'default':
                        cap = cv2.VideoCapture(camera_index)
                    elif method == 'dshow' and sys.platform == 'win32':
                        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    elif method == 'msmf' and sys.platform == 'win32':
                        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
                    else:
                        continue
                    
                    if cap.isOpened():
                        # Test if we can actually read frames
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            camera_opened = True
                            break
                        else:
                            cap.release()
                except Exception as e:
                    if cap:
                        cap.release()
                    continue
            
            if camera_opened:
                break
        
        if not camera_opened:
            # Provide detailed error message with troubleshooting steps
            with camera_placeholder.container():
                st.error("🚫 Could not access camera for video capture")
                
                st.markdown("""
                ### 🔧 Troubleshooting Steps:
                
                1. **Close other applications** using the camera (Zoom, Teams, Skype, etc.)
                
                2. **Refresh the page** and try again
                
                3. **Check Windows Camera Privacy Settings**:
                   - Open Windows Settings → Privacy → Camera
                   - Ensure "Allow apps to access your camera" is ON
                   - Ensure Python/your browser has permission
                
                4. **Try the manual camera test**:
                """)
                
                if st.button("🎥 Run Camera Test", key="manual_camera_test"):
                    # Clear the camera_tested flag to retry
                    if 'camera_tested' in st.session_state:
                        del st.session_state.camera_tested
                    st.rerun()
                
                st.markdown("""
                5. **Alternative: Use the test script**:
                   ```bash
                   python test_camera.py
                   ```
                   
                If the test script works but this doesn't, it may be a Streamlit-specific issue.
                """)
                
                # Show system info for debugging
                with st.expander("🔍 System Information"):
                    st.code(f"""
Platform: {sys.platform}
Python: {sys.version}
OpenCV: {cv2.__version__}
Available backends: {[cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]}
                    """)
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
        
        # Warm up the camera
        for _ in range(5):
            cap.read()
        
        try:
            frame_count = 0
            while st.session_state.tracking_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, tracking_info = self.face_tracker.process_frame(frame)
                
                # Display frame
                with camera_placeholder.container():
                    st.image(processed_frame, channels="BGR", use_column_width=True)
                
                # Update status
                self.update_status(status_placeholder, tracking_info)
                
                # Update analytics every 5 frames to reduce computational load
                if frame_count % 5 == 0:
                    self.update_analytics(analytics_placeholder)
                
                frame_count += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def update_status(self, placeholder, tracking_info):
        """Update real-time status display with dynamic UI"""
        with placeholder.container():
            # Create dynamic status cards
            st.markdown("""
            <style>
                .status-card {
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    transition: all 0.3s ease;
                }
                
                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 0.5rem;
                    animation: pulse 2s infinite;
                }
                
                .status-active {
                    background: #2ecc71;
                    box-shadow: 0 0 10px #2ecc71;
                }
                
                .status-warning {
                    background: #f39c12;
                    box-shadow: 0 0 10px #f39c12;
                }
                
                .status-error {
                    background: #e74c3c;
                    box-shadow: 0 0 10px #e74c3c;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Face Detection Status
            face_status = "status-active" if tracking_info['face_detected'] else "status-error"
            st.markdown(f"""
            <div class="status-card">
                <span class="status-indicator {face_status}"></span>
                <strong>Face Detection:</strong> {'Detected ✓' if tracking_info['face_detected'] else 'Not Detected ✗'}
            </div>
            """, unsafe_allow_html=True)
            
            # Centering Status
            center_status = "status-active" if tracking_info['is_centered'] else "status-warning"
            st.markdown(f"""
            <div class="status-card">
                <span class="status-indicator {center_status}"></span>
                <strong>Position:</strong> {'Centered 🎯' if tracking_info['is_centered'] else 'Off-Center ⚠️'}
            </div>
            """, unsafe_allow_html=True)
            
            # Eye Detection with visual bars
            left_detected = tracking_info.get('left_eye_detected', False)
            right_detected = tracking_info.get('right_eye_detected', False)
            
            st.markdown("""
            <div class="status-card">
                <strong>👁️ Eye Detection</strong>
                <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                    <div style="flex: 1;">
                        <div style="background: """ + ("#2ecc71" if left_detected else "#e74c3c") + """; 
                                    height: 8px; 
                                    border-radius: 4px;
                                    box-shadow: 0 0 10px """ + ("#2ecc71" if left_detected else "#e74c3c") + """;">
                        </div>
                        <small style="color: #888;">Left Eye</small>
                    </div>
                    <div style="flex: 1;">
                        <div style="background: """ + ("#2ecc71" if right_detected else "#e74c3c") + """; 
                                    height: 8px; 
                                    border-radius: 4px;
                                    box-shadow: 0 0 10px """ + ("#2ecc71" if right_detected else "#e74c3c") + """;">
                        </div>
                        <small style="color: #888;">Right Eye</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Looking at Camera Status
            looking_status = "status-active" if tracking_info['looking_at_camera'] else "status-warning"
            st.markdown(f"""
            <div class="status-card">
                <span class="status-indicator {looking_status}"></span>
                <strong>Gaze:</strong> {'Looking at Camera ✅' if tracking_info['looking_at_camera'] else 'Not Looking at Camera ↗️'}
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics with modern design
            col1, col2 = st.columns(2)
            
            with col1:
                # Distance metric with progress bar
                distance = tracking_info['center_distance']
                max_distance = 200  # Maximum expected distance
                progress = min(100, (distance / max_distance) * 100)
                color = "#2ecc71" if distance < self.face_tracker.center_threshold else "#e74c3c"
                
                st.markdown(f"""
                <div class="status-card">
                    <strong>Distance from Center</strong>
                    <div style="margin: 0.5rem 0;">
                        <div style="background: rgba(255, 255, 255, 0.1); 
                                    height: 20px; 
                                    border-radius: 10px; 
                                    overflow: hidden;">
                            <div style="background: {color}; 
                                        width: {progress}%; 
                                        height: 100%;
                                        transition: width 0.3s ease;
                                        box-shadow: 0 0 10px {color};">
                            </div>
                        </div>
                        <small style="color: #888;">{distance:.1f} pixels</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Blink counter with animation
                blinks = tracking_info.get('total_blinks', 0)
                st.markdown(f"""
                <div class="status-card">
                    <strong>Total Blinks</strong>
                    <div style="font-size: 2rem; 
                                font-weight: bold; 
                                color: #667eea;
                                text-align: center;
                                margin: 0.5rem 0;
                                text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);">
                        {blinks}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Eye Aspect Ratios with visual indicators
            if left_detected or right_detected:
                st.markdown("""<div class="status-card"><strong>📊 Eye Metrics</strong></div>""", unsafe_allow_html=True)
                
                ear_col1, ear_col2 = st.columns(2)
                
                with ear_col1:
                    if left_detected:
                        left_ear = tracking_info.get('left_ear', 0)
                        ear_percent = (left_ear / 0.5) * 100  # Assuming max EAR of 0.5
                        color = "#2ecc71" if left_ear > self.face_tracker.EAR_THRESHOLD else "#e74c3c"
                        
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="position: relative; 
                                        width: 80px; 
                                        height: 80px; 
                                        margin: 0 auto;">
                                <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                          fill="none"
                                          stroke="rgba(255, 255, 255, 0.1)"
                                          stroke-width="3"></path>
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                          fill="none"
                                          stroke="{color}"
                                          stroke-width="3"
                                          stroke-dasharray="{ear_percent}, 100"
                                          style="filter: drop-shadow(0 0 5px {color});"></path>
                                </svg>
                                <div style="position: absolute; 
                                            top: 50%; 
                                            left: 50%; 
                                            transform: translate(-50%, -50%);
                                            font-weight: bold;
                                            color: {color};">
                                    {left_ear:.2f}
                                </div>
                            </div>
                            <small style="color: #888;">Left EAR</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with ear_col2:
                    if right_detected:
                        right_ear = tracking_info.get('right_ear', 0)
                        ear_percent = (right_ear / 0.5) * 100
                        color = "#2ecc71" if right_ear > self.face_tracker.EAR_THRESHOLD else "#e74c3c"
                        
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="position: relative; 
                                        width: 80px; 
                                        height: 80px; 
                                        margin: 0 auto;">
                                <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                          fill="none"
                                          stroke="rgba(255, 255, 255, 0.1)"
                                          stroke-width="3"></path>
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                          fill="none"
                                          stroke="{color}"
                                          stroke-width="3"
                                          stroke-dasharray="{ear_percent}, 100"
                                          style="filter: drop-shadow(0 0 5px {color});"></path>
                                </svg>
                                <div style="position: absolute; 
                                            top: 50%; 
                                            left: 50%; 
                                            transform: translate(-50%, -50%);
                                            font-weight: bold;
                                            color: {color};">
                                    {right_ear:.2f}
                                </div>
                            </div>
                            <small style="color: #888;">Right EAR</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Head Orientation with visual gauges
            if tracking_info['face_detected']:
                angles = tracking_info['head_angles']
                
                st.markdown("""<div class="status-card"><strong>🔄 Head Orientation</strong></div>""", unsafe_allow_html=True)
                
                # Create angle visualization
                angle_cols = st.columns(3)
                angle_names = ["Pitch", "Yaw", "Roll"]
                angle_thresholds = [self.face_tracker.pitch_threshold, self.face_tracker.yaw_threshold, 180]
                
                for i, (col, angle, name, threshold) in enumerate(zip(angle_cols, angles, angle_names, angle_thresholds)):
                    with col:
                        normalized = (abs(angle) / threshold) * 100
                        color = "#2ecc71" if abs(angle) < threshold else "#e74c3c"
                        
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <strong style="color: #888;">{name}</strong>
                            <div style="margin: 0.5rem 0; 
                                        font-size: 1.5rem; 
                                        font-weight: bold;
                                        color: {color};
                                        text-shadow: 0 0 10px {color};">
                                {angle:.1f}°
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.1); 
                                        height: 5px; 
                                        border-radius: 3px; 
                                        overflow: hidden;">
                                <div style="background: {color}; 
                                            width: {min(100, normalized)}%; 
                                            height: 100%;
                                            box-shadow: 0 0 5px {color};">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    def update_analytics(self, placeholder):
        """Update analytics display with enhanced visualizations"""
        if len(self.face_tracker.tracking_data) < 2:
            return
        
        with placeholder.container():
            # Convert tracking data to DataFrame
            df_data = []
            for data in list(self.face_tracker.tracking_data)[-60:]:  # Last 2 seconds
                df_data.append({
                    'timestamp': data['timestamp'],
                    'face_detected': data['face_detected'],
                    'is_centered': data['is_centered'],
                    'looking_at_camera': data['looking_at_camera'],
                    'center_distance': data['center_distance'],
                    'left_eye_detected': data.get('left_eye_detected', False),
                    'right_eye_detected': data.get('right_eye_detected', False),
                    'left_ear': data.get('left_ear', 0),
                    'right_ear': data.get('right_ear', 0),
                    'is_blinking': data.get('is_blinking', False)
                })
            
            df = pd.DataFrame(df_data)
            
            if not df.empty:
                # Calculate percentages
                face_detection_rate = df['face_detected'].mean() * 100
                centered_rate = df['is_centered'].mean() * 100
                attention_rate = df['looking_at_camera'].mean() * 100
                both_eyes_rate = (df['left_eye_detected'] & df['right_eye_detected']).mean() * 100
                
                # Display metrics with enhanced visuals
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.05); 
                           backdrop-filter: blur(10px); 
                           border-radius: 15px; 
                           padding: 1rem; 
                           margin-bottom: 1rem;
                           border: 1px solid rgba(255, 255, 255, 0.1);">
                    <strong style="color: white;">📊 Performance Metrics</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Create circular progress indicators
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = [
                    (col1, "Face Detection", face_detection_rate, "#3498db"),
                    (col2, "Centered", centered_rate, "#2ecc71"),
                    (col3, "Attention", attention_rate, "#f39c12"),
                    (col4, "Both Eyes", both_eyes_rate, "#9b59b6")
                ]
                
                for col, label, value, color in metrics:
                    with col:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="position: relative; width: 100px; height: 100px; margin: 0 auto;">
                                <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                          fill="none"
                                          stroke="rgba(255, 255, 255, 0.1)"
                                          stroke-width="3"></path>
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                          fill="none"
                                          stroke="{color}"
                                          stroke-width="3"
                                          stroke-dasharray="{value}, 100"
                                          style="filter: drop-shadow(0 0 10px {color});
                                                 animation: pulse 2s ease-in-out infinite;"></path>
                                </svg>
                                <div style="position: absolute; 
                                            top: 50%; 
                                            left: 50%; 
                                            transform: translate(-50%, -50%);">
                                    <div style="font-size: 1.5rem; 
                                                font-weight: bold; 
                                                color: {color};
                                                text-shadow: 0 0 10px {color};">
                                        {value:.0f}%
                                    </div>
                                </div>
                            </div>
                            <p style="color: #888; margin-top: 0.5rem; font-size: 0.9rem;">{label}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Eye tracking chart with modern styling
                if len(df) > 1 and (df['left_ear'].sum() > 0 or df['right_ear'].sum() > 0):
                    fig_eyes = go.Figure()
                    
                    # Add traces with gradient effect
                    fig_eyes.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['left_ear'],
                        mode='lines',
                        name='Left Eye',
                        line=dict(color='#3498db', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(52, 152, 219, 0.1)'
                    ))
                    
                    fig_eyes.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['right_ear'],
                        mode='lines',
                        name='Right Eye',
                        line=dict(color='#2ecc71', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(46, 204, 113, 0.1)'
                    ))
                    
                    # Add threshold line with animation
                    fig_eyes.add_hline(
                        y=self.face_tracker.EAR_THRESHOLD,
                        line_dash="dash",
                        line_color="#e74c3c",
                        line_width=2,
                        annotation_text="Blink Threshold",
                        annotation_position="top right",
                        annotation_font_color="#e74c3c"
                    )
                    
                    fig_eyes.update_layout(
                        title={
                            'text': 'Eye Aspect Ratio (EAR) Over Time',
                            'font': {'color': 'white', 'size': 16}
                        },
                        xaxis_title='Time',
                        yaxis_title='EAR Value',
                        height=250,
                        margin=dict(l=0, r=0, t=40, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'color': 'white'},
                        yaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'color': 'white'},
                        showlegend=True,
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_eyes, use_container_width=True, config={'displayModeBar': False})
                
                # Distance chart with enhanced styling
                if len(df) > 1:
                    fig_dist = go.Figure()
                    
                    # Create gradient effect for distance
                    fig_dist.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['center_distance'],
                        mode='lines',
                        name='Distance',
                        line=dict(
                            color='#667eea',
                            width=3
                        ),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.1)'
                    ))
                    
                    # Add threshold area
                    fig_dist.add_hrect(
                        y0=0, y1=self.face_tracker.center_threshold,
                        fillcolor="rgba(46, 204, 113, 0.1)",
                        layer="below",
                        line_width=0,
                    )
                    
                    fig_dist.add_hline(
                        y=self.face_tracker.center_threshold,
                        line_dash="dash",
                        line_color="#f39c12",
                        line_width=2,
                        annotation_text="Center Threshold",
                        annotation_position="top right",
                        annotation_font_color="#f39c12"
                    )
                    
                    fig_dist.update_layout(
                        title={
                            'text': 'Distance from Center Over Time',
                            'font': {'color': 'white', 'size': 16}
                        },
                        xaxis_title='Time',
                        yaxis_title='Distance (px)',
                        height=200,
                        margin=dict(l=0, r=0, t=40, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'color': 'white'},
                        yaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'color': 'white'},
                        showlegend=False,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
    
    def save_session_analytics(self):
        """Save current session analytics when tracking stops"""
        if len(self.face_tracker.tracking_data) > 0:
            # Calculate session statistics
            df = pd.DataFrame(list(self.face_tracker.tracking_data))
            
            session_duration = time.time() - st.session_state.get('tracking_start_time', time.time())
            
            # Calculate both eyes rate properly
            both_eyes_rate = 0
            if 'left_eye_detected' in df.columns and 'right_eye_detected' in df.columns:
                both_eyes_rate = (df['left_eye_detected'] & df['right_eye_detected']).mean() * 100
            
            session_stats = {
                'timestamp': datetime.now(),
                'duration_seconds': session_duration,
                'total_frames': len(df),
                'face_detection_rate': df['face_detected'].mean() * 100 if 'face_detected' in df.columns else 0,
                'centered_rate': df['is_centered'].mean() * 100 if 'is_centered' in df.columns else 0,
                'attention_rate': df['looking_at_camera'].mean() * 100 if 'looking_at_camera' in df.columns else 0,
                'both_eyes_rate': both_eyes_rate,
                'total_blinks': self.face_tracker.total_blinks,
                'avg_distance_from_center': df['center_distance'].mean() if 'center_distance' in df.columns else 0,
                'raw_data': df.to_dict('records')
            }
            
            st.session_state.session_analytics.append(session_stats)
            st.session_state.tracking_history.extend(list(self.face_tracker.tracking_data))
            
            # Reset face tracker data
            self.face_tracker.tracking_data.clear()
            self.face_tracker.total_blinks = 0
    
    def show_session_summary(self):
        """Display summary of the last session"""
        if st.session_state.session_analytics:
            last_session = st.session_state.session_analytics[-1]
            
            # Create metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Duration",
                    f"{last_session['duration_seconds']:.1f}s",
                    help="Total tracking duration"
                )
                st.metric(
                    "Total Frames",
                    f"{last_session['total_frames']}",
                    help="Number of frames processed"
                )
            
            with col2:
                st.metric(
                    "Face Detection",
                    f"{last_session['face_detection_rate']:.1f}%",
                    help="Percentage of frames with face detected"
                )
                st.metric(
                    "Both Eyes",
                    f"{last_session['both_eyes_rate']:.1f}%",
                    help="Percentage with both eyes detected"
                )
            
            with col3:
                st.metric(
                    "Centered",
                    f"{last_session['centered_rate']:.1f}%",
                    help="Percentage of time face was centered"
                )
                st.metric(
                    "Attention",
                    f"{last_session['attention_rate']:.1f}%",
                    help="Percentage looking at camera"
                )
            
            with col4:
                st.metric(
                    "Total Blinks",
                    f"{last_session['total_blinks']}",
                    help="Number of blinks detected"
                )
                st.metric(
                    "Avg Distance",
                    f"{last_session['avg_distance_from_center']:.1f}px",
                    help="Average distance from center"
                )
            
            # Show detailed charts
            if st.checkbox("Show Detailed Charts"):
                df = pd.DataFrame(last_session['raw_data'])
                
                # Time series chart
                fig_timeline = go.Figure()
                
                fig_timeline.add_trace(go.Scatter(
                    x=df.index,
                    y=df['center_distance'],
                    mode='lines',
                    name='Distance from Center',
                    line=dict(color='blue', width=2)
                ))
                
                fig_timeline.update_layout(
                    title='Session Timeline',
                    xaxis_title='Frame',
                    yaxis_title='Distance (px)',
                    height=300
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    def export_tracking_data(self):
        """Export tracking data to CSV"""
        if len(self.face_tracker.tracking_data) > 0:
            df_export = pd.DataFrame(list(self.face_tracker.tracking_data))
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Current Session",
                data=csv,
                file_name=f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif st.session_state.tracking_history:
            df_export = pd.DataFrame(st.session_state.tracking_history)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download All History",
                data=csv,
                file_name=f"tracking_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No tracking data available to export")
    
    def run(self):
        """Main application runner"""
        self.initialize_session_state()
        
        # Camera permission check
        if not st.session_state.camera_permission:
            self.request_camera_permission()
            return
        
        # Skip the camera test for now and go directly to dashboard
        # The camera will be tested when the user clicks "Start"
        self.create_dashboard()

# Additional utility functions
def create_requirements_file():
    """Create requirements.txt content"""
    requirements = """
streamlit>=1.28.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
    """
    return requirements.strip()

def setup_instructions():
    """Setup instructions for the application"""
    instructions = """
# Face Tracking Dashboard Setup Instructions

## Installation

1. Create a virtual environment:
```bash
python -m venv face_tracking_env
source face_tracking_env/bin/activate  # On Windows: face_tracking_env\\Scripts\\activate
```

2. Install required packages:
```bash
pip install streamlit opencv-python mediapipe numpy pandas plotly
```

3. Run the application:
```bash
streamlit run face_tracking_dashboard.py
```

## Features

- **Real-time face detection** using MediaPipe
- **Center positioning detection** with visual feedback
- **Gaze tracking** to determine if user is looking at camera
- **Head pose estimation** with pitch, yaw, roll angles
- **Live analytics** with performance metrics
- **User authentication** system
- **Camera permission handling**

## Usage

1. Enter username and password to authenticate
2. Grant camera permission when prompted
3. Click "Start Tracking" to begin monitoring
4. The system will show:
   - Live camera feed with face landmarks
   - Real-time status indicators
   - Performance analytics and metrics

## Customization

- Adjust `center_threshold` in FaceTracker class to change centering sensitivity
- Modify head pose thresholds for gaze detection
- Add additional authentication methods as needed
- Extend analytics with more detailed reporting

## Troubleshooting

- Ensure camera permissions are granted in browser
- Check that no other applications are using the camera
- Verify all dependencies are properly installed
- For better performance, ensure good lighting conditions
"""
    return instructions

# Main execution
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()