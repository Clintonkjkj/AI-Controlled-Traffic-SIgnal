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

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

app.config['DEBUG'] = False
app.config['ENV'] = 'production'

# Constants
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
FRAME_SIZE = (480, 480)  # Reduced frame size
BASE_GREEN_TIME = 10
EXTRA_TIME_PER_VEHICLE = 0.5
SIGNAL_ALLOCATION_INTERVAL = 1
STATE_FILE = "state.json"

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
                "status": "ready"
            }
        
        if updates:
            if 'vehicle_count' in updates:
                vehicle_counts = processing_status[processor_id].get("vehicle_counts", [])
                vehicle_counts.append(updates['vehicle_count'])
                if len(vehicle_counts) > 5:
                    vehicle_counts.pop(0)
                updates['smoothed_vehicle_count'] = sum(vehicle_counts) / len(vehicle_counts)
            
            processing_status[processor_id].update(updates)
        
        save_state(state)

def resize_frame(frame):
    """Resize frame to target dimensions"""
    return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)

def perform_initial_analysis(video_path, processor_id):
    """Analyze first few frames to determine vehicle count"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    vehicle_counts = []
    try:
        for _ in range(10):  # Analyze first 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = resize_frame(frame)
            with model_lock:
                results = model(frame)
            
            detections = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
            vehicle_count = sum(1 for obj in detections if obj in ["car", "truck", "bus", "motorcycle"])
            vehicle_counts.append(vehicle_count)
        
        avg_vehicles = sum(vehicle_counts) / len(vehicle_counts) if vehicle_counts else 0
        return avg_vehicles
    
    finally:
        cap.release()

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
                if last_frame:
                    return {
                        "processor_id": processor_id,
                        "image": last_frame,
                        "signal": "red",
                        "paused": True,
                        "status": "paused",
                        "remaining_green_time": 0
                    }
                return None

            # Check if processing is stopped/paused
            if processor_status.get('status') == 'stopped' or not processor_status.get('processing', True):
                last_frame = processor_status.get("last_frame")
                if last_frame:
                    return {
                        "processor_id": processor_id,
                        "image": last_frame,
                        "signal": "red",
                        "paused": True,
                        "status": "paused",
                        "remaining_green_time": 0
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
        
        vehicle_types = ["car", "truck", "bus", "motorcycle"]
        vehicle_count = sum(counter[obj] for obj in vehicle_types if obj in counter)
        
        emergency_types = ["ambulance", "fire truck", "police car"]
        emergency_detected = any(obj in counter for obj in emergency_types)
        
        top_detections = dict(counter.most_common(5))

        # Prepare frame for display
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate remaining green time
        remaining_green_time = 0
        if current_signal == "green":
            vehicle_count = processing_status[processor_id].get("smoothed_vehicle_count", 0)
            total_green_time = BASE_GREEN_TIME + (vehicle_count * EXTRA_TIME_PER_VEHICLE)
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

        # Update processing status
        with status_lock:
            if processor_id in processing_status:
                processing_status[processor_id].update({
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

    # Perform initial analysis
    try:
        avg_vehicles = perform_initial_analysis(video_path, processor_id)
        update_processing_status(processor_id, {
            "processing": True,
            "complete": False,
            "signal": "red",
            "paused": True,
            "total_frames": total_frames,
            "fps": fps,
            "smoothed_vehicle_count": avg_vehicles,
            "status": "analyzing"
        })
        update_status(processor_id, f"Initial analysis: {avg_vehicles:.1f} avg vehicles")
    except Exception as e:
        emit('status_update', {
            "processor_id": processor_id,
            "message": f"Initial analysis failed: {str(e)}",
            "error": True,
            "status": "error"
        })
        return

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
    """Optimized signal allocation with vehicle-based priority"""
    global last_green_signal, green_start_time, green_signal_order

    with status_lock:
        active_processors = {
            pid: status for pid, status in processing_status.items()
            if status.get('processing', False) and not status.get('complete', False)
        }

        if not active_processors:
            return

        current_time = time.time()
        elapsed_time = current_time - green_start_time

        # Check if current green signal still has time
        if last_green_signal in active_processors:
            vehicle_count = active_processors[last_green_signal].get("smoothed_vehicle_count", 0)
            green_time = BASE_GREEN_TIME + (vehicle_count * EXTRA_TIME_PER_VEHICLE)
            if elapsed_time < green_time:
                # Emit remaining time update
                remaining_time = green_time - elapsed_time
                socketio.emit('signal_time_update', {
                    "processor_id": last_green_signal,
                    "remaining_time": remaining_time,
                    "total_time": green_time
                })
                return  # Keep the current green signal active

        # Find the next signal in round-robin order
        next_green = None

        if last_green_signal in green_signal_order:
            current_idx = green_signal_order.index(last_green_signal)
            for offset in range(1, len(green_signal_order)):
                next_idx = (current_idx + offset) % len(green_signal_order)
                candidate = green_signal_order[next_idx]
                if candidate in active_processors:
                    next_green = candidate
                    break

        # If no suitable candidate found, pick the one with most vehicles
        if not next_green and active_processors:
            next_green = max(
                active_processors.items(),
                key=lambda x: x[1].get("smoothed_vehicle_count", 0)
            )[0]

        if not next_green:
            return

        # First, set all signals to red
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

        # Set the selected processor to green
        processing_status[next_green].update({
            "signal": "green",
            "paused": False
        })

        vehicle_count = active_processors[next_green].get("smoothed_vehicle_count", 0)
        green_time = BASE_GREEN_TIME + (vehicle_count * EXTRA_TIME_PER_VEHICLE)

        socketio.emit('signal_update', {
            "processor_id": next_green,
            "signal": "green",
            "paused": False,
            "green_time": green_time,
            "vehicle_count": vehicle_count
        })

        last_green_signal = next_green
        green_start_time = current_time
        save_state(state)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(allocate_traffic_signals, 'interval', seconds=SIGNAL_ALLOCATION_INTERVAL)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    with app.app_context():
        socketio.run(app, host="127.0.0.1", port=5000, debug=False)