# import asyncio
# import cv2
# import struct
# import time
# import json
# import logging
# from datetime import datetime
# from typing import Optional, Dict, Any
# import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class MiVOLOAnalyticsClient:
#     def __init__(self, server_ip='localhost', server_port=12346, camera_source=0):
#         self.server_ip = server_ip
#         self.server_port = server_port
#         self.camera_source = camera_source
#         self.cap = None
#         self.session_id = None
#         self.frame_count = 0
#         self.analytics_data = []
#         self.display_analytics = True
        
#         # Performance tracking
#         self.fps_counter = 0
#         self.fps_start_time = time.time()
#         self.last_analytics = None
        
#     def initialize_camera(self) -> bool:
#         """Initialize camera capture"""
#         try:
#             self.cap = cv2.VideoCapture(self.camera_source)
#             if not self.cap.isOpened():
#                 logger.error(f"Failed to open camera source: {self.camera_source}")
#                 return False
            
#             # Optimize camera settings for real-time processing
#             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             self.cap.set(cv2.CAP_PROP_FPS, 30)
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
#             # Get actual camera settings
#             width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = self.cap.get(cv2.CAP_PROP_FPS)
            
#             logger.info(f"Camera initialized: {width}x{height} @ {fps} FPS")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error initializing camera: {e}")
#             return False
    
#     def draw_analytics_overlay(self, frame: np.ndarray, analytics: Dict[str, Any]) -> np.ndarray:
#         """Draw analytics information overlay on frame"""
#         if not analytics:
#             return frame
        
#         overlay_frame = frame.copy()
#         height, width = overlay_frame.shape[:2]
        
#         # Create semi-transparent overlay
#         overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
#         # Analytics panel background
#         panel_height = 200
#         cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        
#         # Blend overlay with frame
#         alpha = 0.7
#         cv2.addWeighted(overlay_frame, 1 - alpha, overlay, alpha, 0, overlay_frame)
        
#         # Text properties
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.6
#         color = (255, 255, 255)
#         thickness = 1
        
#         # Draw analytics text
#         y_offset = 35
#         line_height = 20
        
#         texts = [
#             f"Frame: {analytics.get('frame_number', 0)}",
#             f"Processing: {analytics.get('processing_time_ms', 0):.1f}ms",
#             f"Total Persons: {analytics.get('total_persons', 0)}",
#             f"Total Faces: {analytics.get('total_faces', 0)}",
#             f"Males: {analytics.get('male_count', 0)} | Females: {analytics.get('female_count', 0)}",
#             f"Children: {analytics.get('children_count', 0)} | Young: {analytics.get('young_adults_count', 0)}",
#             f"Middle: {analytics.get('middle_aged_count', 0)} | Seniors: {analytics.get('seniors_count', 0)}",
#             f"Unknown Age: {analytics.get('unknown_age_count', 0)} | Unknown Gender: {analytics.get('unknown_gender_count', 0)}"
#         ]
        
#         for i, text in enumerate(texts):
#             y_pos = y_offset + (i * line_height)
#             cv2.putText(overlay_frame, text, (20, y_pos), font, font_scale, color, thickness)
        
#         # Draw bounding boxes for person detections
#         for person in analytics.get('person_detections', []):
#             bbox = person.get('bbox', [0, 0, 0, 0])
#             if len(bbox) >= 4:
#                 # Convert normalized coordinates to pixel coordinates
#                 x1 = int(bbox[0] * width) if bbox[0] <= 1 else int(bbox[0])
#                 y1 = int(bbox[1] * height) if bbox[1] <= 1 else int(bbox[1])
#                 x2 = int((bbox[0] + bbox[2]) * width) if bbox[2] <= 1 else int(bbox[0] + bbox[2])
#                 y2 = int((bbox[1] + bbox[3]) * height) if bbox[3] <= 1 else int(bbox[1] + bbox[3])
                
#                 # Draw person bounding box
#                 cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
#                 # Draw person info
#                 age = person.get('age', 'Unknown')
#                 gender = person.get('gender', 'U')
#                 gender_text = {'M': 'Male', 'F': 'Female', 'U': 'Unknown'}.get(gender, 'Unknown')
                
#                 info_text = f"{gender_text}, Age: {age}"
#                 cv2.putText(overlay_frame, info_text, (x1, y1 - 10), font, 0.5, (0, 255, 0), 1)
        
#         # Draw bounding boxes for face detections
#         for face in analytics.get('face_detections', []):
#             bbox = face.get('bbox', [0, 0, 0, 0])
#             if len(bbox) >= 4:
#                 # Convert normalized coordinates to pixel coordinates
#                 x1 = int(bbox[0] * width) if bbox[0] <= 1 else int(bbox[0])
#                 y1 = int(bbox[1] * height) if bbox[1] <= 1 else int(bbox[1])
#                 x2 = int((bbox[0] + bbox[2]) * width) if bbox[2] <= 1 else int(bbox[0] + bbox[2])
#                 y2 = int((bbox[1] + bbox[3]) * height) if bbox[3] <= 1 else int(bbox[1] + bbox[3])
                
#                 # Draw face bounding box
#                 cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#         return overlay_frame
    
#     def calculate_fps(self) -> Optional[float]:
#         """Calculate and return current FPS"""
#         self.fps_counter += 1
        
#         if self.fps_counter >= 30:  # Calculate FPS every 30 frames
#             elapsed_time = time.time() - self.fps_start_time
#             current_fps = self.fps_counter / elapsed_time
            
#             self.fps_counter = 0
#             self.fps_start_time = time.time()
            
#             return current_fps
        
#         return None
    
#     async def send_frames_and_receive_analytics(self):
#         """Main loop to send frames and receive analytics"""
#         while True:
#             try:
#                 logger.info(f"Connecting to server at {self.server_ip}:{self.server_port}...")
#                 reader, writer = await asyncio.open_connection(self.server_ip, self.server_port)
#                 logger.info("Connected to analytics server")
                
#                 # Setup display window
#                 cv2.namedWindow('MiVOLO Analytics Client', cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow('MiVOLO Analytics Client', 800, 600)
                
#                 while True:
#                     # Capture frame
#                     ret, frame = self.cap.read()
#                     if not ret:
#                         logger.warning("Failed to capture frame")
#                         break
                    
#                     self.frame_count += 1
                    
#                     # Compress frame for transmission
#                     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#                     frame_data = buffer.tobytes()
                    
#                     # Send frame to server
#                     frame_size = struct.pack(">I", len(frame_data))
#                     writer.write(frame_size + frame_data)
#                     await writer.drain()
                    
#                     # Receive analytics response
#                     try:
#                         response_size_data = await asyncio.wait_for(reader.readexactly(4), timeout=5.0)
#                         response_size = struct.unpack(">I", response_size_data)[0]
                        
#                         response_data = await asyncio.wait_for(reader.readexactly(response_size), timeout=5.0)
#                         response_json = json.loads(response_data.decode())
                        
#                         if response_json.get('status') == 'success':
#                             analytics = response_json.get('analytics', {})
#                             self.last_analytics = analytics
#                             self.session_id = response_json.get('session_id')
                            
#                             # Store analytics data
#                             self.analytics_data.append(analytics)
                            
#                             # Keep only last 1000 analytics records in memory
#                             if len(self.analytics_data) > 1000:
#                                 self.analytics_data.pop(0)
                    
#                     except asyncio.TimeoutError:
#                         logger.warning("Timeout waiting for server response")
#                         self.last_analytics = None
                    
#                     # Display frame with analytics overlay
#                     if self.display_analytics and self.last_analytics:
#                         display_frame = self.draw_analytics_overlay(frame, self.last_analytics)
#                     else:
#                         display_frame = frame
                    
#                     cv2.imshow('MiVOLO Analytics Client', display_frame)
                    
#                     # Calculate and display FPS
#                     current_fps = self.calculate_fps()
#                     if current_fps:
#                         logger.info(f"Client FPS: {current_fps:.1f}")
                    
#                     # Handle key presses
#                     key = cv2.waitKey(1) & 0xFF
#                     if key == ord('q'):
#                         logger.info("Quit requested by user")
#                         return
#                     elif key == ord('a'):
#                         self.display_analytics = not self.display_analytics
#                         logger.info(f"Analytics overlay: {'ON' if self.display_analytics else 'OFF'}")
#                     elif key == ord('s'):
#                         self.save_analytics_summary()
                    
#                     # Control frame rate
#                     await asyncio.sleep(1/30)  # 30 FPS
                
#             except ConnectionRefusedError:
#                 logger.error("Connection refused. Server might be down. Retrying in 5 seconds...")
#                 await asyncio.sleep(5)
#             except Exception as e:
#                 logger.error(f"Connection error: {e}. Retrying in 5 seconds...")
#                 await asyncio.sleep(5)
#             finally:
#                 try:
#                     if 'writer' in locals():
#                         writer.close()
#                         await writer.wait_closed()
#                 except Exception as e:
#                     logger.error(f"Error closing connection: {e}")
    
#     def save_analytics_summary(self):
#         """Save analytics summary to file"""
#         if not self.analytics_data:
#             logger.warning("No analytics data to save")
#             return
        
#         try:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"analytics_summary_{timestamp}.json"
            
#             summary = {
#                 'session_id': self.session_id,
#                 'total_frames': len(self.analytics_data),
#                 'start_time': self.analytics_data[0].get('timestamp') if self.analytics_data else None,
#                 'end_time': self.analytics_data[-1].get('timestamp') if self.analytics_data else None,
#                 'analytics_data': self.analytics_data
#             }
            
#             with open(filename, 'w') as f:
#                 json.dump(summary, f, indent=2)
            
#             logger.info(f"Analytics summary saved to {filename}")
            
#         except Exception as e:
#             logger.error(f"Error saving analytics summary: {e}")
    
#     async def run(self):
#         """Run the analytics client"""
#         if not self.initialize_camera():
#             return
        
#         try:
#             logger.info("Starting MiVOLO Analytics Client")
#             logger.info("Controls: 'q' = quit, 'a' = toggle analytics overlay, 's' = save summary")
            
#             await self.send_frames_and_receive_analytics()
            
#         except KeyboardInterrupt:
#             logger.info("Client interrupted by user")
#         finally:
#             if self.cap:
#                 self.cap.release()
#             cv2.destroyAllWindows()
#             logger.info("Client shutdown complete")

# # Main function
# async def main():
#     # Configuration
#     # SERVER_IP = 'localhost'  # Change to server IP address
#     SERVER_IP = '127.0.0.1'  # Change to server IP address

#     SERVER_PORT = 12346
#     # CAMERA_SOURCE = 0  # 0 for default camera, or path to video file
#     CAMERA_SOURCE = "http://192.168.16.107:8080/video?x.mjpg"  # 0 for default camera, or path to video file

    
#     client = MiVOLOAnalyticsClient(SERVER_IP, SERVER_PORT, CAMERA_SOURCE)
#     await client.run()

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import cv2
import struct
import time
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MiVOLOAnalyticsClient:
    def __init__(self, server_ip='localhost', server_port=12346, camera_source=0):
        self.server_ip = server_ip
        self.server_port = server_port
        self.camera_source = camera_source
        self.cap = None
        self.session_id = None
        self.frame_count = 0
        self.analytics_data = []
        self.display_analytics = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.last_analytics = None
        
        # Connection state
        self.connected = False
        self.connection_retry_delay = 5
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture with better error handling"""
        try:
            logger.info(f"Initializing camera source: {self.camera_source}")
            self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.camera_source}")
                return False
            
            # Set camera properties with error checking
            properties = [
                (cv2.CAP_PROP_FRAME_WIDTH, 640),
                (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                (cv2.CAP_PROP_FPS, 30),
                (cv2.CAP_PROP_BUFFERSIZE, 1)
            ]
            
            for prop, value in properties:
                try:
                    self.cap.set(prop, value)
                except Exception as e:
                    logger.warning(f"Failed to set camera property {prop}: {e}")
            
            # Get actual camera settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {width}x{height} @ {fps} FPS")
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info(f"Test frame captured successfully: {test_frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def draw_analytics_overlay(self, frame: np.ndarray, analytics: Dict[str, Any]) -> np.ndarray:
        """Draw analytics information overlay on frame with improved visualization"""
        if not analytics:
            return frame
        
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Analytics panel background - make it adaptive to content
        panel_height = 250
        panel_width = 450
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        
        # Blend overlay with frame
        alpha = 0.8
        cv2.addWeighted(overlay_frame, 1 - alpha, overlay, alpha, 0, overlay_frame)
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green text
        thickness = 1
        
        # Draw analytics text
        y_offset = 35
        line_height = 22
        
        texts = [
            f"Frame: {analytics.get('frame_number', 0)} | Session: {self.session_id[:8] if self.session_id else 'N/A'}",
            f"Processing Time: {analytics.get('processing_time_ms', 0):.1f}ms",
            f"Total Persons: {analytics.get('total_persons', 0)} | Total Faces: {analytics.get('total_faces', 0)}",
            f"Males: {analytics.get('male_count', 0)} | Females: {analytics.get('female_count', 0)} | Unknown: {analytics.get('unknown_gender_count', 0)}",
            f"Children: {analytics.get('children_count', 0)} | Young Adults: {analytics.get('young_adults_count', 0)}",
            f"Middle-aged: {analytics.get('middle_aged_count', 0)} | Seniors: {analytics.get('seniors_count', 0)}",
            f"Unknown Age: {analytics.get('unknown_age_count', 0)}",
            f"Connection: {'Connected' if self.connected else 'Disconnected'}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = y_offset + (i * line_height)
            # Add text shadow for better readability
            cv2.putText(overlay_frame, text, (21, y_pos + 1), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(overlay_frame, text, (20, y_pos), font, font_scale, color, thickness)
        
        # Draw bounding boxes for person detections
        for idx, person in enumerate(analytics.get('person_detections', [])):
            bbox = person.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                # Convert normalized coordinates to pixel coordinates
                x1 = int(bbox[0] * width) if bbox[0] <= 1 else int(bbox[0])
                y1 = int(bbox[1] * height) if bbox[1] <= 1 else int(bbox[1])
                x2 = int((bbox[0] + bbox[2]) * width) if bbox[2] <= 1 else int(bbox[0] + bbox[2])
                y2 = int((bbox[1] + bbox[3]) * height) if bbox[3] <= 1 else int(bbox[1] + bbox[3])
                
                # Ensure coordinates are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Draw person bounding box with different colors for gender
                gender = person.get('gender', 'unknown').lower()
                if gender == 'male':
                    bbox_color = (255, 0, 0)  # Blue for male
                elif gender == 'female':
                    bbox_color = (255, 0, 255)  # Magenta for female
                else:
                    bbox_color = (0, 255, 0)  # Green for unknown
                
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), bbox_color, 2)
                
                # Draw person info
                age = person.get('age', 'Unknown')
                gender_text = gender.capitalize() if gender != 'unknown' else 'Unknown'
                confidence = person.get('confidence', 0)
                
                info_text = f"{gender_text}, {age}"
                if isinstance(age, (int, float)):
                    info_text = f"{gender_text}, {age:.0f}y"
                
                # Background for text
                text_size = cv2.getTextSize(info_text, font, 0.5, 1)[0]
                cv2.rectangle(overlay_frame, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), bbox_color, -1)
                cv2.putText(overlay_frame, info_text, (x1 + 2, y1 - 8), font, 0.5, (255, 255, 255), 1)
                
                # Draw confidence as a small bar
                conf_width = int((x2 - x1) * confidence)
                cv2.rectangle(overlay_frame, (x1, y2 + 2), (x1 + conf_width, y2 + 6), bbox_color, -1)
        
        # Draw bounding boxes for face detections
        for face in analytics.get('face_detections', []):
            bbox = face.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                # Convert normalized coordinates to pixel coordinates
                x1 = int(bbox[0] * width) if bbox[0] <= 1 else int(bbox[0])
                y1 = int(bbox[1] * height) if bbox[1] <= 1 else int(bbox[1])
                x2 = int((bbox[0] + bbox[2]) * width) if bbox[2] <= 1 else int(bbox[0] + bbox[2])
                y2 = int((bbox[1] + bbox[3]) * height) if bbox[3] <= 1 else int(bbox[1] + bbox[3])
                
                # Ensure coordinates are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Draw face bounding box
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow for faces
        
        # Draw FPS counter
        current_fps = self.calculate_fps()
        if current_fps:
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(overlay_frame, fps_text, (width - 120, 30), font, 0.6, (0, 255, 255), 2)
        
        return overlay_frame
    
    def calculate_fps(self) -> Optional[float]:
        """Calculate and return current FPS"""
        self.fps_counter += 1
        
        if self.fps_counter >= 30:  # Calculate FPS every 30 frames
            elapsed_time = time.time() - self.fps_start_time
            if elapsed_time > 0:
                current_fps = self.fps_counter / elapsed_time
                
                self.fps_counter = 0
                self.fps_start_time = time.time()
                
                return current_fps
        
        return None
    
    async def connect_to_server(self):
        """Connect to server with retry logic"""
        while not self.connected:
            try:
                logger.info(f"Connecting to server at {self.server_ip}:{self.server_port}...")
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.server_ip, self.server_port), 
                    timeout=10.0
                )
                logger.info("Connected to analytics server")
                self.connected = True
                return reader, writer
                
            except asyncio.TimeoutError:
                logger.error("Connection timeout. Retrying...")
                await asyncio.sleep(self.connection_retry_delay)
            except ConnectionRefusedError:
                logger.error("Connection refused. Server might be down. Retrying...")
                await asyncio.sleep(self.connection_retry_delay)
            except Exception as e:
                logger.error(f"Connection error: {e}. Retrying...")
                await asyncio.sleep(self.connection_retry_delay)
    
    async def send_frames_and_receive_analytics(self):
        """Main loop to send frames and receive analytics with improved error handling"""
        reader, writer = None, None
        
        try:
            # Setup display window
            cv2.namedWindow('MiVOLO Analytics Client', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MiVOLO Analytics Client', 1024, 768)
            
            while True:
                # Connect to server if not connected
                if not self.connected:
                    try:
                        reader, writer = await self.connect_to_server()
                    except Exception as e:
                        logger.error(f"Failed to connect to server: {e}")
                        # Show disconnected frame
                        ret, frame = self.cap.read()
                        if ret:
                            disconnected_frame = self.draw_disconnected_overlay(frame)
                            cv2.imshow('MiVOLO Analytics Client', disconnected_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            return
                        continue
                
                try:
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to capture frame")
                        await asyncio.sleep(0.1)
                        continue
                    
                    self.frame_count += 1
                    
                    # Compress frame for transmission with better quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    frame_data = buffer.tobytes()
                    
                    # Send frame to server
                    frame_size = struct.pack(">I", len(frame_data))
                    writer.write(frame_size + frame_data)
                    await writer.drain()
                    
                    # Receive analytics response
                    try:
                        response_size_data = await asyncio.wait_for(reader.readexactly(4), timeout=10.0)
                        response_size = struct.unpack(">I", response_size_data)[0]
                        
                        response_data = await asyncio.wait_for(reader.readexactly(response_size), timeout=10.0)
                        response_json = json.loads(response_data.decode())
                        
                        if response_json.get('status') == 'success':
                            analytics = response_json.get('analytics', {})
                            self.last_analytics = analytics
                            self.session_id = response_json.get('session_id')
                            
                            # Store analytics data
                            self.analytics_data.append(analytics)
                            
                            # Keep only last 1000 analytics records in memory
                            if len(self.analytics_data) > 1000:
                                self.analytics_data.pop(0)
                                
                            # Log successful processing
                            if analytics.get('total_persons', 0) > 0 or analytics.get('total_faces', 0) > 0:
                                logger.info(f"Frame {self.frame_count}: {analytics.get('total_persons', 0)} persons, {analytics.get('total_faces', 0)} faces detected")
                        
                        elif response_json.get('status') == 'error':
                            logger.warning(f"Server error: {response_json.get('message', 'Unknown error')}")
                            self.last_analytics = None
                    
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for server response")
                        self.last_analytics = None
                        self.connected = False
                        if writer:
                            writer.close()
                            await writer.wait_closed()
                        continue
                    
                except Exception as e:
                    logger.error(f"Error in frame processing loop: {e}")
                    self.connected = False
                    if writer:
                        try:
                            writer.close()
                            await writer.wait_closed()
                        except:
                            pass
                    continue
                
                # Display frame with analytics overlay
                try:
                    if self.display_analytics and self.last_analytics:
                        display_frame = self.draw_analytics_overlay(frame, self.last_analytics)
                    else:
                        display_frame = frame
                    
                    cv2.imshow('MiVOLO Analytics Client', display_frame)
                except Exception as e:
                    logger.error(f"Error displaying frame: {e}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    return
                elif key == ord('a'):
                    self.display_analytics = not self.display_analytics
                    logger.info(f"Analytics overlay: {'ON' if self.display_analytics else 'OFF'}")
                elif key == ord('s'):
                    self.save_analytics_summary()
                elif key == ord('r'):
                    logger.info("Reconnection requested by user")
                    self.connected = False
                    if writer:
                        try:
                            writer.close()
                            await writer.wait_closed()
                        except:
                            pass
                
                # Control frame rate
                await asyncio.sleep(1/30)  # 30 FPS
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if writer:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
    
    def draw_disconnected_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw disconnected status overlay"""
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Semi-transparent red overlay
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[:, :] = (0, 0, 255)  # Red
        cv2.addWeighted(overlay_frame, 0.8, overlay, 0.2, 0, overlay_frame)
        
        # Disconnected message
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "DISCONNECTED - Trying to reconnect..."
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        cv2.putText(overlay_frame, text, (text_x, text_y), font, 1.0, (255, 255, 255), 2)
        
        return overlay_frame
    
    def save_analytics_summary(self):
        """Save analytics summary to file with better formatting"""
        if not self.analytics_data:
            logger.warning("No analytics data to save")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_summary_{timestamp}.json"
            
            # Calculate summary statistics
            total_persons_detected = sum(frame.get('total_persons', 0) for frame in self.analytics_data)
            total_faces_detected = sum(frame.get('total_faces', 0) for frame in self.analytics_data)
            frames_with_detections = sum(1 for frame in self.analytics_data if frame.get('total_persons', 0) > 0)
            
            # Age and gender statistics
            all_ages = []
            all_genders = []
            for frame in self.analytics_data:
                for person in frame.get('person_detections', []):
                    if person.get('age') is not None:
                        all_ages.append(person['age'])
                    if person.get('gender'):
                        all_genders.append(person['gender'])
            
            summary = {
                'session_info': {
                    'session_id': self.session_id,
                    'total_frames_processed': len(self.analytics_data),
                    'frames_with_detections': frames_with_detections,
                    'start_time': self.analytics_data[0].get('timestamp') if self.analytics_data else None,
                    'end_time': self.analytics_data[-1].get('timestamp') if self.analytics_data else None,
                    'camera_source': str(self.camera_source)
                },
                'detection_summary': {
                    'total_person_detections': total_persons_detected,
                    'total_face_detections': total_faces_detected,
                    'average_persons_per_frame': total_persons_detected / len(self.analytics_data) if self.analytics_data else 0,
                    'average_faces_per_frame': total_faces_detected / len(self.analytics_data) if self.analytics_data else 0
                },
                'demographics': {
                    'ages': {
                        'count': len(all_ages),
                        'average': sum(all_ages) / len(all_ages) if all_ages else 0,
                        'min': min(all_ages) if all_ages else None,
                        'max': max(all_ages) if all_ages else None
                    },
                    'genders': {
                        'male': all_genders.count('male'),
                        'female': all_genders.count('female'),
                        'unknown': all_genders.count('unknown')
                    }
                },
                'raw_analytics_data': self.analytics_data
            }
            
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Analytics summary saved to {filename}")
            logger.info(f"Summary: {len(self.analytics_data)} frames, {total_persons_detected} person detections, {total_faces_detected} face detections")
            
        except Exception as e:
            logger.error(f"Error saving analytics summary: {e}")
    
    async def run(self):
        """Run the analytics client"""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return
        
        try:
            logger.info("Starting MiVOLO Analytics Client")
            logger.info("Controls:")
            logger.info("  'q' = quit")
            logger.info("  'a' = toggle analytics overlay")
            logger.info("  's' = save summary")
            logger.info("  'r' = reconnect to server")
            
            await self.send_frames_and_receive_analytics()
            
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in client: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Client shutdown complete")

# Main function
async def main():
    # Configuration
    SERVER_IP = '127.0.0.1'  # Change to server IP address
    SERVER_PORT = 12346
    
    # Camera source options:
    # - 0 for default camera
    # - "http://192.168.16.107:8080/video?x.mjpg" for IP camera
    # - "path/to/video.mp4" for video file
    # CAMERA_SOURCE = "http://192.168.16.107:8080/video?x.mjpg"  # IP camera
    CAMERA_SOURCE = 0  # Default camera
    
    client = MiVOLOAnalyticsClient(SERVER_IP, SERVER_PORT, CAMERA_SOURCE)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())