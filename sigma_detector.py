import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def calculate_ear(eye_landmarks):
    # Calculate Eye Aspect Ratio (EAR)
    # Vertical distances
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    h1 = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h1) if h1 > 0 else 0
    return ear

def main():
    print("Initializing Camera for Sigma Detection...")
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Face Mesh indices for eyes (simplified)
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    
    # Mouth indices
    MOUTH_INNER = [78, 81, 13, 311, 308, 315, 14, 87]
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LIP_UP = 13
    LIP_DOWN = 14
    
    # Eyebrows
    RIGHT_BROW_DOWN = [282, 295, 285] # Inner right
    LEFT_BROW_DOWN = [52, 65, 55]     # Inner left
    
    print("Welcome to the Sigma Detector!")
    print("Do the Patrick Bateman stare to raise your score.")

    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1) # Mirror
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(img_rgb)
        
        sigma_score = 0
        message = "NORMAL"
        color = (0, 255, 0)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Extract specific landmarks
                r_eye_pts = [landmarks[i] for i in RIGHT_EYE]
                l_eye_pts = [landmarks[i] for i in LEFT_EYE]
                
                # Calculate EAR (Squint detection)
                r_ear = calculate_ear(r_eye_pts)
                l_ear = calculate_ear(l_eye_pts)
                avg_ear = (r_ear + l_ear) / 2.0
                
                # The "sigma stare" involves a prominent squint. 
                # Normal EAR is ~0.30. Fully closed is ~0.10.
                # Squint is around 0.15 to 0.22
                squint_score = 0
                if 0.14 < avg_ear < 0.22:
                    squint_score = 40  # Perfect squint
                elif 0.22 <= avg_ear < 0.26:
                    squint_score = 20  # Mild squint
                    
                # Calculate Lip Distance (Pursed Lips)
                mouth_h = calculate_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
                mouth_v = calculate_distance(landmarks[LIP_UP], landmarks[LIP_DOWN])
                mar = mouth_v / mouth_h if mouth_h > 0 else 0
                
                # Pursed lips mean small vertical opening (mouth_v) but slightly stretched horizontally
                lip_score = 0
                if mar < 0.10: # Mouth tightly closed
                    lip_score = 30
                elif mar < 0.15:
                    lip_score = 15
                    
                # Eyebrow Furrow
                # We check distance from inner eyebrows to the nose bridge
                nose_bridge = landmarks[8]
                r_brow = landmarks[RIGHT_BROW_DOWN[0]]
                l_brow = landmarks[LEFT_BROW_DOWN[0]]
                brow_dist = (calculate_distance(r_brow, nose_bridge) + calculate_distance(l_brow, nose_bridge)) / 2.0
                
                furrow_score = 0
                if brow_dist < 0.05: # Eyebrows are extremely close to the nose bridge center (lowered)
                    furrow_score = 30
                elif brow_dist < 0.06:
                    furrow_score = 15
                    
                # Total Score
                sigma_score = squint_score + lip_score + furrow_score
                
                # Scale bounds
                if sigma_score > 100: sigma_score = 100
                
                # Draw facial mesh landmarks for visual effect
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                
                # Only draw the tessellation lightly for aesthetics
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        
        # Determine status
        if sigma_score > 80:
            message = "PURE SIGMA"
            color = (0, 0, 255) # Red for peak power
        elif sigma_score > 50:
            message = "GRINDSET DETECTED"
            color = (0, 165, 255) # Orange
        else:
            message = "NPC STATUS"
            color = (255, 0, 0) # Blue
            
        # UI Overlays
        # Score Bar
        bar_x, bar_y = 50, 600
        bar_w, bar_h = 400, 30
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Fill bar
        fill_w = int((sigma_score / 100) * bar_w)
        if fill_w > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        
        # Texts
        cv2.putText(img, f'SIGMA LEVEL: {sigma_score}%', (50, 580), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Giant message if high score
        if sigma_score > 50:
            cv2.putText(img, message, (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 2.5, color, 4)
            
            # Draw letterbox (cinematic bars) for peak sigma feel
            if sigma_score > 80:
                cv2.rectangle(img, (0, 0), (w, 150), (0, 0, 0), -1)
                cv2.rectangle(img, (0, h-150), (w, h), (0, 0, 0), -1)
                cv2.putText(img, "S I G M A  M A L E", (w//2 - 250, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)

        cv2.imshow("Sigma Male Detector", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
