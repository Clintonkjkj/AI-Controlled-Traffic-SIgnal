import eventlet
eventlet.monkey_patch()

import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from collections import Counter
import threading
import logging
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from flask_socketio import SocketIO, emit
import math

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

app.config['DEBUG'] = False
app.config['ENV'] = 'production'

# Constants
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
FRAME_SIZE = (480, 480)  # Reduced frame size
BASE_GREEN_TIME = 10  # Minimum green time in seconds
EXTRA_TIME_PER_VEHICLE = 0.5  # Additional seconds per vehicle
MAX_GREEN_TIME = 60  # Maximum green time in seconds
MIN_GREEN_TIME = 5  # Minimum green time in seconds
SIGNAL_ALLOCATION_INTERVAL = 1  # How often to check signal timing
STATE_FILE = "state.json"
VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle"]
EMERGENCY_TYPES = ["ambulance", "fire truck", "police car"]

# Setup logging
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# Threading setup
executor = ThreadPoolExecutor(max_workers=2)
model_lock = threading.Lock()
status_lock = threading.Lock()
stop_event = threading.Event()

def load_state():
    """Load application state with proper initialization"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                # Clear inactive processors
                state["processing_status"] = {}
                state["green_signal_order"] = []
                state["green_start_time"] = time.time()
                return state
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading state: {e}, using defaults")
    
    return {
        "processing_status": {},
        "last_green_signal": None,
        "green_signal_order": [],
        "green_start_time": time.time(),
    }

def save_state(state):
    """Save application state to file"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# Initialize state
state = load_state()
processing_status = state["processing_status"]
last_green_signal = state["last_green_signal"]
green_signal_order = state["green_signal_order"]
green_start_time = state["green_start_time"]

def generate_processor_id(video_path):
    """Generate unique processor ID from video path"""
    return f"video-{hashlib.md5(video_path.encode()).hexdigest()}"

def update_status(processor_id, status_message, video_path="", progress=None):
    """Update and emit processing status"""
    log_msg = f"Processor {processor_id}: {status_message}"
    logging.info(log_msg)
    socketio.emit('status_update', {
        "processor_id": processor_id,
        "message": status_message,
        "progress": progress,
        "status": "complete" if status_message.lower() == "complete" else "processing"
    })

def update_processing_status(processor_id, updates=None):
    """Update processor status with thread safety"""
    global processing_status
    
    with status_lock:
        if processor_id not in processing_status:
            processing_status[processor_id] = {
                "vehicle_counts": [],
                "last_frame": None,
                "status": "ready",
                "signal": "red",
                "paused": True
            }
        
        if updates:
            if 'vehicle_count' in updates:
                # Convert to int before storing in the list
                try:
                    vehicle_count = max(0, int(round(float(updates['vehicle_count']))))
                except (ValueError, TypeError):
                    vehicle_count = 0
                    
                vehicle_counts = processing_status[processor_id].get("vehicle_counts", [])
                vehicle_counts.append(vehicle_count)
                
                # Keep a reasonable number of readings for smoothing
                if len(vehicle_counts) > 10:  # Keep last 10 readings
                    vehicle_counts.pop(0)
                
                # Calculate smoothed vehicle count
                if vehicle_counts:
                    # Use weighted average where recent counts matter more
                    weights = [i+1 for i in range(len(vehicle_counts))]  # Linear weights
                    weighted_sum = sum(v*w for v,w in zip(vehicle_counts, weights))
                    total_weight = sum(weights)
                    updates['smoothed_vehicle_count'] = max(0, int(round(weighted_sum / total_weight)))
                else:
                    updates['smoothed_vehicle_count'] = 0
            
            processing_status[processor_id].update(updates)
        
        save_state(state)

def resize_frame(frame):
    """Resize frame to target dimensions"""
    return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)

def process_frame(video_path, frame_idx, interval, processor_id):
    """Process a single video frame with enhanced detection"""
    try:
        # First check if processing should stop
        if stop_event.is_set():
            return None

        # Check current signal status
        with status_lock:
            processor_status = processing_status.get(processor_id, {})
            current_signal = processor_status.get("signal", "red")
            
            # If signal is red, return the last frame if available
            if current_signal == "red":
                last_frame = processor_status.get("last_frame")
                last_vehicle_count = processor_status.get("smoothed_vehicle_count", 0)
                if last_frame:
                    return {
                        "processor_id": processor_id,
                        "image": last_frame,
                        "signal": "red",
                        "paused": True,
                        "status": "paused",
                        "remaining_green_time": 0,
                        "vehicles": last_vehicle_count
                    }
                return None

            # Check if processing is stopped/paused
            if processor_status.get('status') == 'stopped' or not processor_status.get('processing', True):
                last_frame = processor_status.get("last_frame")
                last_vehicle_count = processor_status.get("smoothed_vehicle_count", 0)
                if last_frame:
                    return {
                        "processor_id": processor_id,
                        "image": last_frame,
                        "signal": "red",
                        "paused": True,
                        "status": "paused",
                        "remaining_green_time": 0,
                        "vehicles": last_vehicle_count
                    }
                return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * interval)
        ret, frame = cap.read()
        if not ret:
            update_processing_status(processor_id, {"complete": True, "status": "complete"})
            update_status(processor_id, "Complete")
            return None

        frame = resize_frame(frame)
        
        # Perform object detection
        with model_lock:
            results = model(frame)
            annotated_frame = results[0].plot()

        # Count vehicles and detect emergency vehicles
        detections = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        counter = Counter(detections)
        
        vehicle_count = int(sum(counter[obj] for obj in VEHICLE_TYPES if obj in counter))
        emergency_detected = any(obj in counter for obj in EMERGENCY_TYPES)
        top_detections = dict(counter.most_common(5))

        # Prepare frame for display
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate remaining green time
        remaining_green_time = 0
        if current_signal == "green":
            total_green_time = min(MAX_GREEN_TIME, 
                                 max(MIN_GREEN_TIME, 
                                     BASE_GREEN_TIME + (vehicle_count * EXTRA_TIME_PER_VEHICLE)))
            elapsed_time = time.time() - green_start_time
            remaining_green_time = max(0, total_green_time - elapsed_time)

        response = {
            "processor_id": processor_id,
            "image": frame_base64,
            "objects": top_detections,
            "vehicles": vehicle_count,
            "emergency": emergency_detected,
            "signal": current_signal,
            "paused": False,
            "status": "processing",
            "frame_idx": frame_idx,
            "remaining_green_time": remaining_green_time
        }

        # Update processing status with the latest data
        update_processing_status(processor_id, {
            "last_frame": frame_base64,
            "vehicle_count": vehicle_count,
            "emergency": emergency_detected,
            "last_update": time.time(),
            "frame_data": response
        })

        return response

    except Exception as e:
        logging.error(f"Error processing frame: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    files = request.files.getlist("videos")
    file_paths = []
    for file in files:
        if file.filename:
            filename = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            file_paths.append(file_path)
    return jsonify({"message": "Files uploaded successfully", "files": file_paths})

@socketio.on('process_video')
def handle_process_video(data):
    """Handle video processing request"""
    video_path = data['video_path']
    interval = data['interval']
    processor_id = data['processor_id']

    if not os.path.exists(video_path):
        emit('status_update', {
            "processor_id": processor_id,
            "message": f"Video not found: {video_path}",
            "error": True,
            "status": "error"
        })
        return

    # Clear stop event when starting new processing
    stop_event.clear()

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Initialize processing status
    update_processing_status(processor_id, {
        "processing": True,
        "complete": False,
        "signal": "red",
        "paused": True,
        "total_frames": total_frames,
        "fps": fps,
        "smoothed_vehicle_count": 0,  # Start with 0, will be updated in real-time
        "status": "ready"
    })

    # Add to processing queue
    if processor_id not in green_signal_order:
        green_signal_order.append(processor_id)
        save_state(state)

    # Start processing
    socketio.start_background_task(
        process_video_stream, 
        video_path, interval, processor_id, total_frames
    )

def process_video_stream(video_path, interval, processor_id, total_frames):
    """Process video stream with signal control"""
    def process_next_frame(frame_idx):
        # Check stop condition before processing each frame
        if stop_event.is_set():
            update_processing_status(processor_id, {"complete": True, "status": "stopped"})
            update_status(processor_id, "Processing stopped by user")
            return

        with status_lock:
            processor_status = processing_status.get(processor_id, {})
            if processor_status.get('status') == 'stopped' or not processor_status.get('processing', True):
                update_processing_status(processor_id, {"complete": True, "status": "stopped"})
                update_status(processor_id, "Processing stopped")
                return

        if frame_idx * interval >= total_frames:
            update_processing_status(processor_id, {"complete": True, "status": "complete"})
            update_status(processor_id, "Complete")
            return

        frame_data = process_frame(video_path, frame_idx, interval, processor_id)
        if frame_data:
            socketio.emit('frame_update', frame_data)
            # Only increment frame index if the signal is green
            with status_lock:
                if processing_status.get(processor_id, {}).get("signal") == "green":
                    frame_idx += 1
        else:
            # If no frame data (red signal), keep the same frame index
            pass

        eventlet.sleep(0.1)
        socketio.start_background_task(process_next_frame, frame_idx)

    socketio.start_background_task(process_next_frame, 0)

@socketio.on('stop_all_processing')
def handle_stop_processing():
    """Stop all processing tasks effectively"""
    global processing_status
    stop_event.set()
    
    with status_lock:
        for pid in list(processing_status.keys()):
            processing_status[pid].update({
                'processing': False,
                'complete': True,
                'status': 'stopped',
                'paused': True,
                'signal': 'red'
            })
        
        # Clear processing queue
        green_signal_order.clear()
        save_state(state)
    
    # Notify all clients that processing has stopped
    emit('processing_stopped', {'message': 'All processing stopped'}, broadcast=True)
    logging.info("All processing stopped by user")

def allocate_traffic_signals():
    """Optimized signal allocation with vehicle-based priority and proper switching"""
    global last_green_signal, green_start_time, green_signal_order

    with status_lock:
        # Get active processors (those currently processing)
        active_processors = {
            pid: status for pid, status in processing_status.items()
            if status.get('processing', False) and not status.get('complete', False)
        }

        if not active_processors:
            return

        current_time = time.time()
        elapsed_time = current_time - green_start_time

        # Check if current green signal is still active and has time remaining
        if last_green_signal in active_processors:
            processor_status = active_processors[last_green_signal]
            vehicle_count = processor_status.get("smoothed_vehicle_count", 0)
            
            # Calculate dynamic green time based on current vehicle count
            green_time = calculate_green_time(vehicle_count)
            
            if elapsed_time < green_time:
                # Emit remaining time update
                remaining_time = green_time - elapsed_time
                socketio.emit('signal_time_update', {
                    "processor_id": last_green_signal,
                    "remaining_time": remaining_time,
                    "total_time": green_time,
                    "vehicle_count": vehicle_count
                })
                return  # Keep current green signal active

        # If we get here, it's time to switch signals
        
        # 1. First set all signals to red
        for pid in active_processors:
            processing_status[pid].update({
                "signal": "red",
                "paused": True
            })
            socketio.emit('signal_update', {
                "processor_id": pid,
                "signal": "red",
                "paused": True,
                "green_time": 0
            })

        # 2. Select the next signal to turn green
        next_green = select_next_signal(active_processors)
        if not next_green:
            return  # No suitable signal found

        # 3. Get vehicle count for the selected signal
        processor_status = active_processors[next_green]
        vehicle_count = processor_status.get("smoothed_vehicle_count", 0)
        green_time = calculate_green_time(vehicle_count)

        # 4. Update the selected processor to green
        processing_status[next_green].update({
            "signal": "green",
            "paused": False
        })

        logging.info(f"Switching signal to green for {next_green} with {vehicle_count} vehicles for {green_time} seconds")
        
        # 5. Emit updates to clients
        socketio.emit('signal_update', {
            "processor_id": next_green,
            "signal": "green",
            "paused": False,
            "green_time": green_time,
            "vehicle_count": vehicle_count
        })

        # 6. Update global state
        last_green_signal = next_green
        green_start_time = current_time
        save_state(state)

def calculate_green_time(vehicle_count):
    """Calculate appropriate green time based on vehicle count"""
    return min(MAX_GREEN_TIME, 
              max(MIN_GREEN_TIME, 
                  BASE_GREEN_TIME + (vehicle_count * EXTRA_TIME_PER_VEHICLE)))

def select_next_signal(active_processors):
    """
    Select the next signal to turn green based on:
    1. Emergency vehicles detected
    2. Vehicle counts
    3. Round-robin fairness
    """
    # First check for emergency vehicles
    emergency_signals = [
        pid for pid, status in active_processors.items()
        if status.get('emergency', False)
    ]
    
    if emergency_signals:
        return emergency_signals[0]  # Prioritize first emergency
    
    # If no emergencies, use vehicle count + round-robin
    if not green_signal_order:
        green_signal_order.extend(active_processors.keys())
    
    # Get the current position in the round-robin order
    current_pos = 0
    if last_green_signal in green_signal_order:
        current_pos = green_signal_order.index(last_green_signal)
    
    # Find next suitable signal in order
    for offset in range(1, len(green_signal_order) + 1):
        next_idx = (current_pos + offset) % len(green_signal_order)
        candidate = green_signal_order[next_idx]
        
        if candidate in active_processors:
            return candidate
    
    # Fallback to highest vehicle count if round-robin fails
    try:
        return max(
            active_processors.items(),
            key=lambda x: x[1].get("smoothed_vehicle_count", 0)
        )[0]
    except:
        return None

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(allocate_traffic_signals, 'interval', seconds=SIGNAL_ALLOCATION_INTERVAL)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    with app.app_context():
        socketio.run(app, host="127.0.0.1", port=5000, debug=False)
