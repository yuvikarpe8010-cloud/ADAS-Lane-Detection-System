# Advanced Driver Assistance System (ADAS)

A comprehensive ADAS implementation combining edge-based lane detection, object detection, and autonomous steering control using OpenCV and YOLOv8.

## Features

### Core Functionality
- **Edge-based Lane Detection**: Canny edge detection + Hough Transform for accurate lane line detection
- **Region of Interest (ROI)**: Focuses on bottom 40% of frame for lane detection
- **Lane Center Calculation**: Computes lane center from detected left and right lane lines
- **PD Controller Steering**: Proportional-Derivative controller for smooth steering based on lane center error
- **Real-time Object Detection**: Detects persons and vehicles using YOLOv8
- **Time-to-Collision (TTC)**: Calculates collision risk based on object movement
- **Risk Assessment**: Color-coded risk levels (SAFE/WARNING/CRITICAL)
- **Decision Making**: Autonomous driving actions (DRIVING/BRAKE/STEER LEFT/RIGHT)

### Visual Features
- **Professional UI Overlay**: Clean ADAS dashboard with real-time metrics
- **Lane Line Visualization**: Green lines showing detected lane boundaries
- **Center Line Indicators**: Blue frame center line and yellow lane center line
- **Steering Arrow**: Dynamic steering direction indicator (only shown when steering)
- **Bounding Boxes**: Object detection with class names and distance estimates
- **Color-coded Alerts**: Green (Safe), Orange (Warning), Red (Critical)

### Debug Windows
- **Camera View**: Main ADAS dashboard output
- **Lane Edges ROI**: Black background with white edge detection results
- **Debug Stats**: Press 'd' to toggle detailed statistics window

### Performance & Control
- **FPS Counter**: Real-time performance monitoring
- **PD Controller**: Tunable KP/KD parameters for steering control
- **Optimized Processing**: YOLOv8n with 256x256 input for speed
- **Modular Architecture**: Separate functions for detection, steering, and visualization

## Requirements

- Python 3.8+
- ultralytics (YOLOv8)
- OpenCV
- NumPy
- Webcam/Camera

## Installation

1. Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

2. Download YOLOv8 model:
```bash
# The script will automatically download yolov8n.pt if not present
```

## Usage

Run the ADAS system:
```bash
python lane.py
```

### Controls
- **ESC**: Exit the application
- **'d'**: Toggle debug statistics window

## Technical Details

### Lane Detection Pipeline
1. Convert frame to grayscale
2. Apply Gaussian blur (5x5 kernel)
3. Canny edge detection (low=50, high=150)
4. Apply trapezoidal ROI (bottom 40% of frame)
5. Hough Line Transform to detect lane lines
6. Separate left/right lines based on slope
7. Calculate lane center as midpoint of averaged lines

### Steering Control
- **Error Calculation**: `error = lane_center - frame_center`
- **PD Controller**: `steering = KP * error + KD * derivative`
- **Normalization**: Clamped to ±30 degrees
- **Smoothing**: Exponential smoothing for stable control

### Visualization
- Green lane lines on camera feed
- Blue center line (frame center)
- Yellow lane center line
- Red steering arrow (when active)
- Black "Lane Edges ROI" window with white edges
```

### Controls
- **ESC**: Exit the application
- **'d'**: Toggle debug statistics window

## System Architecture

### Modular Functions
- `detect_objects()`: Object detection and lane classification
- `compute_ttc()`: Time-to-Collision calculation
- `assess_risk()`: Risk level determination
- `decide_action()`: Driving decision logic
- `control_speed()`: Speed control simulation
- `compute_steering()`: Steering angle calculation

### Visual Components
- `draw_lane_regions()`: Lane boundary visualization
- `draw_steering_visualization()`: Steering indicators
- `draw_ui_overlay()`: Main UI overlay
- `draw_debug_stats()`: Debug information window

## Technical Details

### Distance Estimation
Uses bounding box area for rough distance approximation:
```
distance = max(1, 50000 / area)
```

### TTC Calculation
Based on rate of change of object area:
```
TTC = current_area / (dA/dt)
```

### Smoothing
- TTC values smoothed over 5-frame history
- Steering angles smoothed with exponential filter
- Speed changes smoothed for realistic behavior

### Lane Classification
Road divided into thirds:
- LEFT: 0-426px
- CENTER: 427-853px
- RIGHT: 854-1280px

## Performance Optimization

- YOLOv8n (lightweight model)
- 256x256 input resolution
- Confidence threshold: 0.5
- Targeted 20+ FPS on modern hardware

## Viva Preparation Notes

### Key Concepts to Explain
1. **Object Detection Pipeline**: YOLOv8 architecture and COCO classes
2. **Lane Detection**: Region-based classification approach
3. **TTC Calculation**: Physics-based collision prediction
4. **Risk Assessment**: Multi-factor decision making
5. **UI Design**: OpenCV drawing functions and transparency

### Code Structure
- Clean separation of concerns with modular functions
- Comprehensive comments for each major section
- Color-coded constants for maintainability
- Global state management for real-time operation

### Algorithm Explanations
- Distance estimation using bounding box area
- Object tracking using position-based IDs
- Smoothing techniques for stable outputs
- Decision logic hierarchy (safety first)

## Troubleshooting

### Common Issues
1. **Camera not found**: Check camera index (default: 0)
2. **Low FPS**: Reduce YOLO input size or confidence threshold
3. **False detections**: Adjust confidence threshold
4. **Import errors**: Ensure all dependencies are installed

### Performance Tuning
- Increase `YOLO_IMGSZ` for accuracy (trade-off with speed)
- Adjust `CONFIDENCE_THRESHOLD` for detection sensitivity
- Modify smoothing parameters for responsiveness vs stability

## Future Enhancements

- Integration with actual vehicle CAN bus
- Advanced lane detection using computer vision
- Integration with GPS for speed validation
- Machine learning-based risk prediction
- Multi-camera support for 360° awareness