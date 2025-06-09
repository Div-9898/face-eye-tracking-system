import cv2
import mediapipe as mp
import numpy as np

def test_camera():
    """Test camera access and basic eye detection"""
    print("Testing camera access...")
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Try to open camera
    cap = None
    for idx in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Successfully opened camera at index {idx}")
                break
        except:
            continue
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Please check:")
        print("1. Camera permissions")
        print("2. No other app is using the camera")
        print("3. Camera drivers are installed")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit, 'c' to capture a frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # Draw simple face detection status
        if results.multi_face_landmarks:
            cv2.putText(frame, "Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Count detected landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite('test_capture.jpg', frame)
            print("Frame captured as test_capture.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Camera test completed")

if __name__ == "__main__":
    test_camera() 