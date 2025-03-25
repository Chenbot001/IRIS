import cv2
import mediapipe as mp
import time
import math
import os

from libgx.libgx11 import SSRHand
from libgx.utils import search_ports


def calc_control_angle(c1, c2, c3):
    """
    Calculate the angle formed at c2 using coordinates of c1, c2, and c3.

    Args:
        c1 (tuple): (x, y, z) coordinates of the first point.
        c2 (tuple): (x, y, z) coordinates of the vertex point.
        c3 (tuple): (x, y, z) coordinates of the third point.

    Returns:
        float: Angle at c2 in degrees.
    """
    # Create vectors
    vector_u = (c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2])  # Vector from c2 to c1
    vector_v = (c3[0] - c2[0], c3[1] - c2[1], c3[2] - c2[2])  # Vector from c2 to c3

    # Calculate dot product
    dot_product = sum(u * v for u, v in zip(vector_u, vector_v))

    # Calculate magnitudes
    magnitude_u = math.sqrt(sum(u ** 2 for u in vector_u))
    magnitude_v = math.sqrt(sum(v ** 2 for v in vector_v))

    # Avoid division by zero
    if magnitude_u == 0 or magnitude_v == 0:
        return 0.0

    # Calculate the angle in radians
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)

    return 180 - angle_degrees # supplementary angle

#---------------------------------------------

# connect to hand hardware
hand = SSRHand(port="COM5")  # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize variables for FPS calculation
prev_time = 0


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break
    
    # Start time for the current frame
    curr_time = time.time()
    
    # Convert the frame to RGB (MediaPipe expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    os.system('cls' if os.name == 'nt' else 'clear')

    control_angles = []

    # Draw hand annotations and calculate supplementary angles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Initialize list to store angles
            control_angles = []

            # Define fingers and corresponding landmark indices for MCP and PIP joints
            fingers = ['Pinky', 'Ring', 'Middle', 'Index']
            mcp_indices = [17,13,9,5]
            pip_indices = [18,14,10,6]
            dip_indices = [19,15,11,7]

            # Calculate MCP and PIP angles for each finger
            for finger, mcp_idx, pip_idx, dip_idx in zip(fingers, mcp_indices, pip_indices, dip_indices):
                # Extract coordinates for MCP angle calculation
                wrist = hand_landmarks.landmark[0]  # Wrist
                mcp = hand_landmarks.landmark[mcp_idx]  # MCP joint
                pip = hand_landmarks.landmark[pip_idx]  # PIP joint
                dip = hand_landmarks.landmark[dip_idx]  # DIP joint

                # Calculate PIP angle
                pip_angle = calc_control_angle(
                    (mcp.x, mcp.y, mcp.z),
                    (pip.x, pip.y, pip.z),
                    (dip.x, dip.y, dip.z)
                )

                control_angles.append(pip_angle)

                # Calculate MCP angle
                mcp_angle = calc_control_angle(
                    (wrist.x, wrist.y, wrist.z),
                    (mcp.x, mcp.y, mcp.z),
                    (pip.x, pip.y, pip.z)
                )
                
                control_angles.append(mcp_angle) #

                # Print angles for debugging
                print(f"{finger} MCP: {mcp_angle:.2f} degrees, {finger} PIP: {pip_angle:.2f} degrees")

            # The control_angles list now contains 8 angles
        
        dof_angle = [0, control_angles[0], control_angles[1], # 小指侧摆（左摆为正，右摆为负）、小指PIP内收、小指MCP内收
                    0, control_angles[2], control_angles[3], # 无名指侧摆（左摆为正，右摆为负）、无名指PIP内收、无名指MCP内收
                    0, control_angles[4], control_angles[5], # 中指侧摆（左摆为正，右摆为负）、中指PIP内收、中指MCP内收
                    0, control_angles[6], control_angles[7], # 食指侧摆（左摆为正，右摆为负）、食指PIP内收、食指MCP内收
                    45, 0, 0] # 拇指侧摆（左摆为正，右摆为负）、拇指PIP内收、拇指MCP内收

        angle = hand.convert_dof_to_motor_angle(dof_angle)

        hand.Safe_control(angle)

    else:
        print("No hands detected.")

    # Calculate FPS
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # Display the FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow("Hand Tracking with MCP Angles", frame) # control approx 20 Hz


    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break




cap.release()
cv2.destroyAllWindows()
hand.off()