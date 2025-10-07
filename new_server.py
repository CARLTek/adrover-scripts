import asyncio
import struct
import numpy as np
import cv2
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import uuid
import torch

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Device Count: {torch.cuda.device_count()}")

# Import MiVOLO components from main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import Predictor

# Simple Config class to hold configuration
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

from mivolo.structures import PersonAndFaceResult

# Django setup for database operations
import django
from django.conf import settings
from django.utils import timezone

# Configure Django settings
if not settings.configured:
    settings.configure(
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': 'camera_analytics.db',
            }
        },
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
            '__main__',  # This module
        ],
        USE_TZ=True,
        TIME_ZONE='UTC',
    )
    django.setup()

# Import Django models (assuming they're in the same directory)
try:
    from django_models import CameraSession, AnalyticsFrame, PersonDetection, FaceDetection, SessionSummary
except ImportError:
    print("Warning: Django models not found. Database operations will be disabled.")
    CameraSession = AnalyticsFrame = PersonDetection = FaceDetection = SessionSummary = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MiVOLOAnalyticsServer:
    def __init__(self, host='localhost', port=12346):
        self.host = host
        self.port = port
        self.predictor = None
        self.active_sessions: Dict[str, dict] = {}
        self.initialize_mivolo()
    
    def initialize_mivolo(self):
        """Initialize MiVOLO predictor with CUDA support"""
        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"PyTorch Version: {torch.__version__}")
            else:
                device = "cpu"
                logger.warning("CUDA is not available. Using CPU.")
            
            # MiVOLO configuration
            args = {
                "detector_weights": "models/yolov8x_person_face.pt",
                "checkpoint": "models/model_imdb_cross_person_4.22_99.46.pth.tar",
                "device": device,  # Use detected device
                "with_persons": True,
                "draw": False,  # Don't draw for server processing
                "disable_faces": False,
                "conf_thresh": 0.3,  # Lowered threshold for better detection
                "iou_thresh": 0.5,   # Lowered threshold for better detection
                "max_det": 50,       # Increased max detections
                "use_grayscale": False,
                "resize_input": False,
                "input_size": 640
            }
            
            config = Config(**args)
            self.predictor = Predictor(config, verbose=True)
            
            # Test CUDA memory if using GPU
            if device == "cuda":
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            logger.info("MiVOLO predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MiVOLO predictor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.predictor = None
    
    def process_frame_with_mivolo(self, frame: np.ndarray, session_id: str, frame_number: int) -> Optional[dict]:
        """Process frame with MiVOLO and return analytics data"""
        if self.predictor is None:
            logger.warning("MiVOLO predictor not available")
            return None
        
        start_time = time.time()
        
        try:
            # Ensure frame is in correct format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB for MiVOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            logger.debug(f"Processing frame {frame_number} with shape: {frame.shape}")
            
            # Run MiVOLO prediction
            detected_objects, _ = self.predictor.recognize(frame_rgb)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.debug(f"MiVOLO detection completed in {processing_time:.2f}ms")
            logger.debug(f"Detected: {detected_objects.n_persons} persons, {detected_objects.n_faces} faces")
            
            # Extract analytics data
            analytics_data = self.extract_analytics_from_results(detected_objects, processing_time, frame_number)
            
            # Store in database if available
            if CameraSession is not None:
                self.store_analytics_in_database(session_id, analytics_data, frame_number)
            else:
                # Store in JSON file
                self.store_analytics_in_json(session_id, analytics_data, frame_number)
            
            return analytics_data
            
        except Exception as e:
            logger.error(f"Error processing frame with MiVOLO: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def extract_analytics_from_results(self, detected_objects: PersonAndFaceResult, processing_time: float, frame_number: int) -> dict:
        """Extract analytics data from MiVOLO results"""
        analytics = {
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': processing_time,
            'total_persons': 0,
            'total_faces': 0,
            'male_count': 0,
            'female_count': 0,
            'unknown_gender_count': 0,
            'children_count': 0,
            'young_adults_count': 0,
            'middle_aged_count': 0,
            'seniors_count': 0,
            'unknown_age_count': 0,
            'person_detections': [],
            'face_detections': []
        }

        try:
            # Get total counts from MiVOLO structure
            analytics['total_persons'] = detected_objects.n_persons if hasattr(detected_objects, 'n_persons') else 0
            analytics['total_faces'] = detected_objects.n_faces if hasattr(detected_objects, 'n_faces') else 0

            logger.debug(f"Raw detection counts: {analytics['total_persons']} persons, {analytics['total_faces']} faces")

            # Check if we have YOLO results
            if not hasattr(detected_objects, 'yolo_results') or detected_objects.yolo_results is None:
                logger.warning("No YOLO results found in detected_objects")
                return analytics

            if not hasattr(detected_objects.yolo_results, 'boxes') or detected_objects.yolo_results.boxes is None:
                logger.warning("No boxes found in YOLO results")
                return analytics

            # Process detections
            boxes = detected_objects.yolo_results.boxes
            if len(boxes) == 0:
                logger.debug("No detections found")
                return analytics

            # Get class names/labels
            classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else []
            
            logger.debug(f"Found {len(boxes)} total detections with classes: {classes}")

            # Process each detection
            for idx in range(len(boxes)):
                try:
                    cls_id = int(classes[idx]) if idx < len(classes) else 0
                    
                    # Get detection data
                    bbox = boxes.xyxy[idx].cpu().numpy() if hasattr(boxes, 'xyxy') else None
                    confidence = float(boxes.conf[idx].cpu().numpy()) if hasattr(boxes, 'conf') else 0.0
                    
                    if bbox is None:
                        continue
                    
                    # Get age and gender predictions
                    age = None
                    gender = 'unknown'
                    gender_score = 0.0
                    
                    if hasattr(detected_objects, 'ages') and idx < len(detected_objects.ages):
                        age = detected_objects.ages[idx]
                    
                    if hasattr(detected_objects, 'genders') and idx < len(detected_objects.genders):
                        gender = detected_objects.genders[idx] if detected_objects.genders[idx] else 'unknown'
                    
                    if hasattr(detected_objects, 'gender_scores') and idx < len(detected_objects.gender_scores):
                        gender_score = detected_objects.gender_scores[idx] if detected_objects.gender_scores[idx] else 0.0

                    # Normalize bbox coordinates
                    if hasattr(detected_objects.yolo_results, 'orig_shape'):
                        orig_h, orig_w = detected_objects.yolo_results.orig_shape
                        bbox_norm = [
                            float(bbox[0] / orig_w),
                            float(bbox[1] / orig_h), 
                            float((bbox[2] - bbox[0]) / orig_w),
                            float((bbox[3] - bbox[1]) / orig_h)
                        ]
                    else:
                        bbox_norm = [float(x) for x in bbox]

                    # Determine age group
                    age_group = 'UNKNOWN'
                    if age is not None:
                        age = float(age)
                        if age <= 17:
                            age_group = 'CHILD'
                        elif age <= 35:
                            age_group = 'YOUNG'
                        elif age <= 55:
                            age_group = 'MIDDLE'
                        else:
                            age_group = 'SENIOR'

                    detection_data = {
                        'bbox': bbox_norm,
                        'age': age,
                        'age_group': age_group,
                        'gender': gender.lower() if gender else 'unknown',
                        'confidence': float(confidence),
                        'gender_confidence': float(gender_score) if gender_score else None,
                        'track_id': None,
                        'class_id': cls_id
                    }

                    # Determine if this is a person or face detection based on class_id or size
                    # Assuming class 0 = person, class 1 = face (adjust based on your model)
                    if cls_id == 0:  # Person detection
                        analytics['person_detections'].append(detection_data)
                        
                        # Update gender counts
                        if gender and gender.lower() == 'male':
                            analytics['male_count'] += 1
                        elif gender and gender.lower() == 'female':
                            analytics['female_count'] += 1
                        else:
                            analytics['unknown_gender_count'] += 1

                        # Update age group counts
                        if age_group == 'CHILD':
                            analytics['children_count'] += 1
                        elif age_group == 'YOUNG':
                            analytics['young_adults_count'] += 1
                        elif age_group == 'MIDDLE':
                            analytics['middle_aged_count'] += 1
                        elif age_group == 'SENIOR':
                            analytics['seniors_count'] += 1
                        else:
                            analytics['unknown_age_count'] += 1
                            
                    else:  # Face detection
                        analytics['face_detections'].append(detection_data)

                except Exception as e:
                    logger.error(f"Error processing detection {idx}: {e}")
                    continue

            logger.debug(f"Processed analytics: {len(analytics['person_detections'])} persons, {len(analytics['face_detections'])} faces")

        except Exception as e:
            logger.error(f"Error extracting analytics from results: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return analytics

    def store_analytics_in_json(self, session_id: str, analytics_data: dict, frame_number: int):
        """Store analytics data in JSON file for easy viewing"""
        try:
            # Create analytics directory if it doesn't exist
            analytics_dir = "analytics_data"
            if not os.path.exists(analytics_dir):
                os.makedirs(analytics_dir)

            # Create session file path
            session_file = os.path.join(analytics_dir, f"session_{session_id[:8]}.json")

            # Load existing data or create new
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            else:
                session_data = {
                    "session_id": session_id,
                    "start_time": datetime.now().isoformat(),
                    "frames": [],
                    "summary": {
                        "total_frames": 0,
                        "total_persons_detected": 0,
                        "total_faces_detected": 0,
                        "gender_distribution": {"male": 0, "female": 0, "unknown": 0},
                        "age_distribution": {"child": 0, "young": 0, "middle": 0, "senior": 0, "unknown": 0},
                        "average_age": 0,
                        "age_range": {"min": None, "max": None}
                    }
                }

            # Add current frame data
            session_data["frames"].append(analytics_data)

            # Update summary statistics
            summary = session_data["summary"]
            summary["total_frames"] += 1

            # Update frame-based statistics
            summary["total_person_detections"] = summary.get("total_person_detections", 0) + analytics_data["total_persons"]
            summary["total_face_detections"] = summary.get("total_face_detections", 0) + analytics_data["total_faces"]

            # Calculate unique persons estimate (simplified approach)
            frames_with_people = sum(1 for frame in session_data["frames"] if frame["total_persons"] > 0)
            if frames_with_people > 0:
                # Estimate unique persons as the most common person count across frames
                person_counts = [frame["total_persons"] for frame in session_data["frames"] if frame["total_persons"] > 0]
                from collections import Counter
                if person_counts:
                    most_common_count = Counter(person_counts).most_common(1)[0][0]
                    summary["estimated_unique_persons"] = most_common_count
                    summary["estimated_unique_faces"] = most_common_count
            else:
                summary["estimated_unique_persons"] = 0
                summary["estimated_unique_faces"] = 0

            # Update demographics based on current frame only if we have detections
            if analytics_data["total_persons"] > 0:
                summary["current_demographics"] = {
                    "male_count": analytics_data["male_count"],
                    "female_count": analytics_data["female_count"],
                    "unknown_gender_count": analytics_data["unknown_gender_count"],
                    "children_count": analytics_data["children_count"],
                    "young_adults_count": analytics_data["young_adults_count"],
                    "middle_aged_count": analytics_data["middle_aged_count"],
                    "seniors_count": analytics_data["seniors_count"],
                    "unknown_age_count": analytics_data["unknown_age_count"]
                }

            # Calculate age statistics
            ages = [p.get('age') for p in analytics_data['person_detections'] if p.get('age') is not None]
            if ages:
                if summary["age_range"]["min"] is None or min(ages) < summary["age_range"]["min"]:
                    summary["age_range"]["min"] = min(ages)
                if summary["age_range"]["max"] is None or max(ages) > summary["age_range"]["max"]:
                    summary["age_range"]["max"] = max(ages)

                # Calculate running average
                all_ages = []
                for frame in session_data["frames"]:
                    frame_ages = [p.get('age') for p in frame['person_detections'] if p.get('age') is not None]
                    all_ages.extend(frame_ages)

                if all_ages:
                    summary["average_age"] = sum(all_ages) / len(all_ages)

            # Save updated data
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

            logger.debug(f"Analytics data saved to {session_file}")

        except Exception as e:
            logger.error(f"Error storing analytics in JSON: {e}")

    def store_analytics_in_database(self, session_id: str, analytics_data: dict, frame_number: int):
        """Store analytics data in Django database and JSON file"""
        try:
            # Store in JSON file for easy viewing
            self.store_analytics_in_json(session_id, analytics_data, frame_number)

            logger.info(f"Frame {frame_number}: {analytics_data['total_persons']} persons, {analytics_data['total_faces']} faces")
            
            # Log detailed detection info
            if analytics_data['person_detections']:
                ages = [p.get('age') for p in analytics_data['person_detections'] if p.get('age') is not None]
                genders = [p.get('gender') for p in analytics_data['person_detections'] if p.get('gender')]
                logger.info(f"Ages detected: {ages}, Genders: {genders}")
            
            # Skip database operations to avoid async context issues
            return

        except Exception as e:
            logger.error(f"Error storing analytics in database: {e}")
    
    async def handle_client(self, reader, writer):
        """Handle client connection and process frames"""
        addr = writer.get_extra_info('peername')
        session_id = str(uuid.uuid4())
        logger.info(f"New client connected from {addr}, session: {session_id}")
        
        # Initialize session data
        self.active_sessions[session_id] = {
            'start_time': time.time(),
            'frame_count': 0,
            'total_persons': 0,
            'total_faces': 0,
            'analytics_history': []
        }
        
        try:
            while True:
                # Receive frame size
                frame_size_data = await reader.readexactly(4)
                frame_size = struct.unpack(">I", frame_size_data)[0]
                
                # Receive frame data
                frame_data = await reader.readexactly(frame_size)
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning(f"Received invalid frame from {addr}")
                    continue
                
                # Process frame with MiVOLO
                frame_number = self.active_sessions[session_id]['frame_count']
                analytics_data = self.process_frame_with_mivolo(frame, session_id, frame_number)
                
                if analytics_data:
                    # Update session statistics
                    session_data = self.active_sessions[session_id]
                    session_data['frame_count'] += 1
                    session_data['total_persons'] += analytics_data['total_persons']
                    session_data['total_faces'] += analytics_data['total_faces']
                    session_data['analytics_history'].append(analytics_data)
                    
                    # Keep only last 100 frames in memory
                    if len(session_data['analytics_history']) > 100:
                        session_data['analytics_history'].pop(0)
                    
                    # Send analytics back to client
                    response_data = {
                        'status': 'success',
                        'session_id': session_id,
                        'analytics': analytics_data
                    }
                else:
                    # Send error response
                    response_data = {
                        'status': 'error',
                        'session_id': session_id,
                        'message': 'Failed to process frame'
                    }
                
                response_json = json.dumps(response_data).encode()
                response_size = struct.pack(">I", len(response_json))
                
                writer.write(response_size + response_json)
                await writer.drain()
                
                if analytics_data:
                    logger.info(f"Processed frame {frame_number}: {analytics_data['total_persons']} persons, {analytics_data['total_faces']} faces")
                
        except asyncio.IncompleteReadError:
            logger.info(f"Client {addr} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Clean up session
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                duration = time.time() - session_data['start_time']
                logger.info(f"Session {session_id} ended. Duration: {duration:.2f}s, Frames: {session_data['frame_count']}")
                del self.active_sessions[session_id]
            
            writer.close()
            await writer.wait_closed()
    
    async def start_server(self):
        """Start the analytics server"""
        try:
            server = await asyncio.start_server(self.handle_client, self.host, self.port)
            logger.info(f"MiVOLO Analytics Server running on {self.host}:{self.port}")
            
            # Print GPU info if available
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            async with server:
                await server.serve_forever()
                
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
            import traceback
            logger.error(traceback.format_exc())

# Main function
async def main():
    # Create database tables if using Django models
    if CameraSession is not None:
        try:
            from django.core.management import execute_from_command_line
            execute_from_command_line(['manage.py', 'migrate', '--run-syncdb'])
            logger.info("Database tables created/updated")
        except Exception as e:
            logger.warning(f"Could not create database tables: {e}")
    
    # Start the server
    server = MiVOLOAnalyticsServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())