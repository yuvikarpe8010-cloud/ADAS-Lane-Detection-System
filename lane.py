"""
Advanced Driver Assistance System (ADAS) using OpenCV and YOLOv8
Features:
- Real-time object detection (persons, vehicles)
- Lane detection and region classification
- Time-to-Collision (TTC) calculation
- Risk assessment and decision making
- Steering and speed control simulation
- Professional UI overlay with color-coded alerts
- FPS monitoring and debug mode
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 20
CONFIDENCE_THRESHOLD = 0.5
YOLO_IMGSZ = 256

# Lane detection parameters
GAUSSIAN_BLUR = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 40
HOUGH_MAX_LINE_GAP = 5
ROI_HEIGHT_RATIO = 0.4  # Bottom 40% of frame

# PD Controller parameters
KP = 0.02  # Proportional gain
KD = 0.01  # Derivative gain

# UI Colors
COLOR_SAFE = (0, 255, 0)      # Green
COLOR_WARNING = (0, 165, 255) # Orange
COLOR_CRITICAL = (0, 0, 255)  # Red
COLOR_NEUTRAL = (255, 255, 255) # White
COLOR_UI_BG = (0, 0, 0)       # Black background for text
COLOR_LANE = (200, 200, 200)  # Gray for lanes
COLOR_LANE_LINES = (0, 255, 0)  # Green for detected lane lines
COLOR_CENTER_LINE = (255, 0, 0)  # Blue for center line
COLOR_LANE_CENTER = (0, 255, 255)  # Yellow for lane center

# Smoothing parameters
TTC_HISTORY_SIZE = 5
STEERING_SMOOTHING = 0.1

# Memory and thresholds
PERSON_MEMORY_TIME = 1.0
TTC_CRITICAL = 3.0
TTC_WARNING = 6.0
MAX_STEERING_ANGLE = 30

# ---------------- GLOBAL STATE ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Check if camera is available, if not use synthetic mode
use_synthetic = not cap.isOpened()
if use_synthetic:
    print("Camera not available. Using synthetic test pattern.")
    print("Press 'ESC' to exit")

# State variables
speed = 40.0
steering_angle = 0.0
throttle = 50
brake = False
person_memory = 0
debug_mode = False

# PD Controller state
prev_error = 0.0

# History for smoothing
ttc_history = deque(maxlen=TTC_HISTORY_SIZE)
prev_steering = 0.0

# Object tracking
prev_areas = {}
prev_times = {}

# FPS calculation
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Lane definitions
LANE_WIDTH = FRAME_WIDTH // 3
LANE_CENTER = FRAME_WIDTH // 2
LANES = {
    "LEFT": (0, LANE_WIDTH),
    "CENTER": (LANE_WIDTH, 2 * LANE_WIDTH),
    "RIGHT": (2 * LANE_WIDTH, FRAME_WIDTH)
}

# Class names (YOLOv8 COCO dataset)
CLASS_NAMES = {
    0: "PERSON",
    2: "VEHICLE", 3: "VEHICLE", 5: "VEHICLE", 7: "VEHICLE"  # car, motorcycle, bus, truck
}

def detect_objects(frame):
    """
    Detect objects using YOLOv8 and classify them by lane
    Returns: dict of lane objects and distances
    """
    results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=YOLO_IMGSZ, verbose=False)

    lane_objects = {"LEFT": "CLEAR", "CENTER": "CLEAR", "RIGHT": "CLEAR"}
    lane_distances = {"LEFT": 999, "CENTER": 999, "RIGHT": 999}
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            area = (x2 - x1) * (y2 - y1)

            # Determine lane
            lane_name = "CENTER"  # default
            for name, (start, end) in LANES.items():
                if start <= cx < end:
                    lane_name = name
                    break

            # Simple distance estimation based on bounding box area
            distance = max(1, 50000 / area)

            # Update lane status
            obj_class = CLASS_NAMES[cls]
            if obj_class == "PERSON":
                lane_objects[lane_name] = "PERSON"
                global person_memory
                person_memory = time.time()
            elif obj_class == "VEHICLE":
                lane_objects[lane_name] = "VEHICLE"

            lane_distances[lane_name] = min(lane_distances[lane_name], distance)

            # Store object for TTC calculation
            detected_objects.append({
                'id': int(cx / 10),  # Simple ID based on position
                'area': area,
                'class': obj_class,
                'bbox': (x1, y1, x2, y2),
                'lane': lane_name,
                'distance': distance
            })

    return lane_objects, lane_distances, detected_objects

def detect_lanes(frame):
    """
    Detect lane lines using Canny edge detection and Hough Transform
    Returns: lane_center, left_lines, right_lines, edges_roi
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # Apply Region of Interest (bottom 40% of frame)
    height, width = edges.shape
    roi_height = int(height * ROI_HEIGHT_RATIO)
    roi_vertices = np.array([[
        (0, height),
        (width * 0.1, height - roi_height),
        (width * 0.9, height - roi_height),
        (width, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    edges_roi = cv2.bitwise_and(edges, mask)

    # Use HoughLinesP to detect lane lines
    lines = cv2.HoughLinesP(edges_roi, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                           minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Separate left and right lines based on slope
            if slope < -0.5:  # Left lane (negative slope)
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:  # Right lane (positive slope)
                right_lines.append((x1, y1, x2, y2))

    # Calculate average left and right lines
    def average_lines(lines):
        if not lines:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        if not x_coords:
            return None
        return np.mean(x_coords), np.mean(y_coords)

    left_avg = average_lines(left_lines)
    right_avg = average_lines(right_lines)

    # Compute lane_center = midpoint of both lines
    lane_center = width // 2  # Default to frame center
    if left_avg and right_avg:
        lane_center = (left_avg[0] + right_avg[0]) // 2
    elif left_avg:
        lane_center = left_avg[0] + width // 6  # Estimate right line
    elif right_avg:
        lane_center = right_avg[0] - width // 6  # Estimate left line

    return int(lane_center), left_lines, right_lines, edges_roi

def compute_ttc(objects, prev_areas, prev_times):
    """
    Compute Time-to-Collision for detected objects
    Returns: minimum TTC value
    """
    min_ttc = 999
    current_time = time.time()

    for obj in objects:
        obj_id = obj['id']

        if obj_id in prev_areas and obj_id in prev_times:
            prev_area = prev_areas[obj_id]
            prev_time = prev_times[obj_id]

            dA = obj['area'] - prev_area
            dt = current_time - prev_time

            if dA > 0 and dt > 0:
                ttc = obj['area'] / (dA / dt)
                min_ttc = min(min_ttc, ttc)

        # Update tracking
        prev_areas[obj_id] = obj['area']
        prev_times[obj_id] = current_time

    return min_ttc

def assess_risk(ttc, person_detected):
    """
    Assess current risk level based on TTC and pedestrian detection
    Returns: risk level string and color
    """
    if ttc < TTC_CRITICAL or person_detected:
        return "CRITICAL", COLOR_CRITICAL
    elif ttc < TTC_WARNING:
        return "WARNING", COLOR_WARNING
    else:
        return "SAFE", COLOR_SAFE

def decide_action(lane_objects, ttc, person_detected):
    """
    Decide the appropriate driving action based on current situation
    Returns: action string
    """
    if person_detected:
        return "BRAKE"
    elif ttc < TTC_CRITICAL:
        return "BRAKE"
    elif lane_objects["CENTER"] != "CLEAR":
        if lane_objects["LEFT"] == "CLEAR":
            return "STEER LEFT"
        elif lane_objects["RIGHT"] == "CLEAR":
            return "STEER RIGHT"

    return "DRIVING"

def control_speed(action, current_speed):
    """
    Control vehicle speed based on action
    Returns: new speed, throttle, brake
    """
    target_speed = 60.0

    if action == "BRAKE":
        target_speed = 0.0
        brake = True
        throttle = 0
    else:
        brake = False
        throttle = 50

    # Smooth speed changes
    new_speed = current_speed + (target_speed - current_speed) * 0.1
    new_speed = max(0, min(80, new_speed))

    return new_speed, throttle, brake

def compute_steering(lane_center, current_steering, prev_error):
    """
    Compute steering angle using PD controller based on lane center
    Returns: new steering angle, updated prev_error
    """
    frame_center = FRAME_WIDTH // 2
    error = lane_center - frame_center

    # PD controller: steering = kp * error + kd * derivative
    derivative = error - prev_error
    steering = KP * error + KD * derivative

    # Normalize steering to range (-30 to +30 degrees)
    steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering))

    # Smooth steering changes
    new_steering = current_steering + (steering - current_steering) * STEERING_SMOOTHING
    new_steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, new_steering))

    return new_steering, error

def draw_lane_regions(frame):
    """
    Draw lane boundaries and labels on the frame
    """
    # Draw lane lines
    cv2.line(frame, (LANE_WIDTH, 0), (LANE_WIDTH, FRAME_HEIGHT), COLOR_LANE, 2)
    cv2.line(frame, (2 * LANE_WIDTH, 0), (2 * LANE_WIDTH, FRAME_HEIGHT), COLOR_LANE, 2)

    # Draw lane labels
    for name, (start, end) in LANES.items():
        center_x = (start + end) // 2
        cv2.putText(frame, name, (center_x - 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NEUTRAL, 2)

def draw_steering_visualization(frame, steering_angle, lane_center):
    """
    Draw center line (blue), lane center (yellow), and steering direction arrow
    """
    car_center_x = FRAME_WIDTH // 2
    car_center_y = FRAME_HEIGHT - 50

    # Draw center line (blue vertical line at frame center)
    cv2.line(frame, (car_center_x, car_center_y - 30), (car_center_x, car_center_y + 30),
             COLOR_CENTER_LINE, 3)

    # Draw lane center (yellow vertical line at lane center)
    cv2.line(frame, (lane_center, car_center_y - 30), (lane_center, car_center_y + 30),
             COLOR_LANE_CENTER, 3)

    # Draw steering direction arrow (only if non-zero)
    if abs(steering_angle) > 1.0:
        arrow_length = 50
        angle_rad = np.radians(steering_angle)
        arrow_end_x = int(car_center_x + arrow_length * np.sin(angle_rad))
        arrow_end_y = int(car_center_y - arrow_length * np.cos(angle_rad))

        cv2.arrowedLine(frame, (car_center_x, car_center_y), (arrow_end_x, arrow_end_y),
                       COLOR_CRITICAL, 3, tipLength=0.3)

def draw_lane_lines(frame, left_lines, right_lines):
    """
    Draw detected lane lines in GREEN on the original frame
    """
    # Draw left lines
    for x1, y1, x2, y2 in left_lines:
        cv2.line(frame, (x1, y1), (x2, y2), COLOR_LANE_LINES, 5)

    # Draw right lines
    for x1, y1, x2, y2 in right_lines:
        cv2.line(frame, (x1, y1), (x2, y2), COLOR_LANE_LINES, 5)

def draw_ui_overlay(frame, stats, speed, steering_angle, ttc, risk, risk_color,
                   action, lane_objects, lane_distances, detected_objects, fps):
    """
    Draw professional UI overlay on the camera feed
    """
    # Semi-transparent background for text overlay
    overlay = frame.copy()

    # Top status bar
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 80), COLOR_UI_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Main metrics
    y_offset = 25
    cv2.putText(frame, f"Speed: {int(speed)} km/h", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NEUTRAL, 2)
    cv2.putText(frame, f"Steering: {int(steering_angle)}°", (200, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NEUTRAL, 2)
    cv2.putText(frame, f"TTC: {ttc:.1f}s", (400, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
    cv2.putText(frame, f"Risk: {risk}", (550, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
    cv2.putText(frame, f"Action: {action}", (750, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
    cv2.putText(frame, f"FPS: {fps}", (FRAME_WIDTH - 100, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NEUTRAL, 2)

    # Draw bounding boxes and labels
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        color = COLOR_CRITICAL if obj['class'] == 'PERSON' else COLOR_WARNING
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with class and distance
        label = f"{obj['class']}: {obj['distance']:.1f}m"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Lane status in bottom left
    cv2.rectangle(frame, (10, FRAME_HEIGHT - 120), (200, FRAME_HEIGHT - 10), COLOR_UI_BG, -1)
    cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)

    y_lane = FRAME_HEIGHT - 100
    cv2.putText(frame, "LANE STATUS:", (20, y_lane),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NEUTRAL, 2)
    for i, (lane, status) in enumerate(lane_objects.items()):
        color = COLOR_CRITICAL if status != "CLEAR" else COLOR_SAFE
        cv2.putText(frame, f"{lane}: {status}", (20, y_lane + 25 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_debug_stats(stats_frame, lane_objects, lane_distances, throttle, brake, ttc_history):
    """
    Draw debug statistics window
    """
    stats_frame[:] = COLOR_UI_BG

    def put_text(text, y, color=COLOR_NEUTRAL):
        cv2.putText(stats_frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    y = 30
    put_text("ADAS DEBUG STATS", y, COLOR_SAFE)
    y += 40

    put_text(f"Throttle: {throttle}%", y)
    y += 30
    put_text(f"Brake: {'ON' if brake else 'OFF'}", y, COLOR_CRITICAL if brake else COLOR_SAFE)
    y += 40

    put_text("LANE DISTANCES:", y, COLOR_WARNING)
    y += 30
    for lane, dist in lane_distances.items():
        put_text(f"{lane}: {dist:.1f}m", y)
        y += 25

    y += 20
    put_text("TTC HISTORY:", y, COLOR_WARNING)
    y += 30
    for i, ttc in enumerate(ttc_history):
        put_text(f"[{i}]: {ttc:.1f}s", y)
        y += 25

# ---------------- MAIN LOOP ----------------
print("Starting ADAS System...")
print("Press 'd' to toggle debug mode, 'ESC' to exit")

# Synthetic frame generator
frame_counter = 0

def generate_synthetic_frame():
    """Generate a synthetic road test pattern"""
    global frame_counter
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Sky (dark gray)
    frame[:FRAME_HEIGHT//2, :] = (40, 40, 50)
    # Road (dark gray)
    frame[FRAME_HEIGHT//2:, :] = (60, 60, 70)
    
    # Road perspective lines (simulating a road)
    horizon = FRAME_HEIGHT // 2
    center_x = FRAME_WIDTH // 2
    
    # Animate lane lines moving
    offset = (frame_counter * 3) % 100
    
    # Left lane line (dashed)
    for i in range(horizon, FRAME_HEIGHT, 20):
        y = i + offset
        if y < FRAME_HEIGHT:
            x_left = int(center_x - 150 * (y - horizon) / (FRAME_HEIGHT - horizon))
            cv2.circle(frame, (x_left, y), 3, (255, 255, 255), -1)
    
    # Right lane line (dashed)
    for i in range(horizon, FRAME_HEIGHT, 20):
        y = i + offset
        if y < FRAME_HEIGHT:
            x_right = int(center_x + 150 * (y - horizon) / (FRAME_HEIGHT - horizon))
            cv2.circle(frame, (x_right, y), 3, (255, 255, 255), -1)
    
    # Center dashed line
    for i in range(horizon, FRAME_HEIGHT, 30):
        y = i + offset
        if y < FRAME_HEIGHT:
            x_center = int(center_x + 20 * np.sin((y - horizon) / 30))
            cv2.circle(frame, (x_center, y), 2, (200, 200, 0), -1)
    
    # Road edge lines (solid)
    cv2.line(frame, (int(center_x - 200), horizon), (int(center_x - 350), FRAME_HEIGHT), (255, 255, 255), 3)
    cv2.line(frame, (int(center_x + 200), horizon), (int(center_x + 350), FRAME_HEIGHT), (255, 255, 255), 3)
    
    # Add some "objects" (rectangles representing cars)
    if frame_counter % 100 < 50:
        # Simulated car ahead
        car_y = horizon + 150
        car_x = center_x - 30
        cv2.rectangle(frame, (car_x - 40, car_y - 25), (car_x + 40, car_y + 25), (0, 100, 200), -1)
        cv2.rectangle(frame, (car_x - 40, car_y - 25), (car_x + 40, car_y + 25), (0, 200, 255), 2)
    
    frame_counter += 1
    return frame

while True:
    if use_synthetic:
        frame = generate_synthetic_frame()
        ret = True
    else:
        ret, frame = cap.read()
    
    if not ret:
        break

    # FPS calculation
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

    # Lane detection
    lane_center, left_lines, right_lines, edges_roi = detect_lanes(frame)

    # Draw detected lane lines in GREEN
    draw_lane_lines(frame, left_lines, right_lines)

    # Object detection
    lane_objects, lane_distances, detected_objects = detect_objects(frame)

    # TTC calculation
    raw_ttc = compute_ttc(detected_objects, prev_areas, prev_times)
    ttc_history.append(raw_ttc)
    smoothed_ttc = sum(ttc_history) / len(ttc_history) if ttc_history else 999

    # Risk assessment
    person_detected = (time.time() - person_memory) < PERSON_MEMORY_TIME
    risk, risk_color = assess_risk(smoothed_ttc, person_detected)

    # Decision making
    action = decide_action(lane_objects, smoothed_ttc, person_detected)

    # Control systems
    speed, throttle, brake = control_speed(action, speed)
    steering_angle, prev_error = compute_steering(lane_center, steering_angle, prev_error)

    # Draw visualizations
    draw_lane_regions(frame)
    draw_steering_visualization(frame, steering_angle, lane_center)

    # Create stats frame for debug
    stats_frame = np.zeros((520, 520, 3), dtype=np.uint8)

    # Draw UI overlay
    draw_ui_overlay(frame, stats_frame, speed, steering_angle, smoothed_ttc,
                   risk, risk_color, action, lane_objects, lane_distances,
                   detected_objects, current_fps)

    # Show camera feed
    cv2.imshow("Camera View", frame)

    # Show Lane Edges ROI window (black background with white edges)
    edges_display = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Lane Edges ROI", edges_display)

    # Show debug stats if enabled
    if debug_mode:
        draw_debug_stats(stats_frame, lane_objects, lane_distances, throttle, brake, list(ttc_history))
        cv2.imshow("ADAS Debug Stats", stats_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('d'):  # Toggle debug mode
        debug_mode = not debug_mode
        if debug_mode:
            print("Debug mode: ON")
        else:
            print("Debug mode: OFF")
            cv2.destroyWindow("ADAS Debug Stats")

cap.release()
cv2.destroyAllWindows()
print("ADAS System stopped.")
