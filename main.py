import cv2
import os
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import torch

# Patch torch.load to use weights_only=False for compatibility with older model files
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    # Force weights_only=False for compatibility with older YOLO models
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

# Apply the patch
torch.load = patched_torch_load

class Predictor:
    def __init__(self, config, verbose: bool = False):
        # Initialize detector with performance optimizations
        conf_thresh = getattr(config, 'conf_thresh', 0.5)
        iou_thresh = getattr(config, 'iou_thresh', 0.7)

        # Enable half precision for GPU, disable for CPU
        use_half = config.device.startswith('cuda')

        self.detector = Detector(
            config.detector_weights,
            config.device,
            half=use_half,  # Enable half precision for GPU performance
            verbose=verbose,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )
        # Enable half precision for GPU, disable for CPU
        use_half = config.device.startswith('cuda')

        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=use_half,  # Enable half precision for GPU performance
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

        # Performance optimization settings
        self.use_grayscale = getattr(config, 'use_grayscale', False)
        self.resize_input = getattr(config, 'resize_input', False)
        self.input_size = getattr(config, 'input_size', 640)

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        # Apply preprocessing optimizations
        processed_image = self._preprocess_image(image)

        detected_objects: PersonAndFaceResult = self.detector.predict(processed_image)
        self.age_gender_model.predict(processed_image, detected_objects)

        out_im = None
        if self.draw:
            out_im = detected_objects.plot()

        return detected_objects, out_im

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing optimizations for faster inference"""
        processed = image.copy()

        # Resize input for faster processing
        if self.resize_input:
            height, width = processed.shape[:2]
            if max(height, width) > self.input_size:
                scale = self.input_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Convert to grayscale if enabled (but keep 3 channels for model compatibility)
        if self.use_grayscale:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return processed

    def recognize_video(self, source: str) -> Generator:
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            for guid, data in cur_persons.items():
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)
            if self.draw:
                frame = detected_objects.plot()
            yield detected_objects_history, frame

# Main script logic
if __name__ == "__main__":
    video_path = "video/video.mp4"
    CAMERA_SOURCE = "http://192.168.16.107:8080/video?x.mjpg"

    # Auto-detect best available device
    def get_best_device():
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("⚠️  No CUDA GPU available, using CPU")
        return device

    device = get_best_device()

    args = {
        "detector_weights": "models/yolov8x_person_face.pt",
        "checkpoint": "models/model_imdb_cross_person_4.22_99.46.pth.tar",
        "device": device,  # Auto-detected device (GPU if available, CPU otherwise)
        "with_persons": True,
        "draw": True,
        "disable_faces": False,
        # Performance optimization settings
        "conf_thresh": 0.6,  # Higher confidence threshold for faster processingC
        "iou_thresh": 0.7,   # IoU threshold for NMS
        "max_det": 30,       # Reduced maximum detections per frame
        # Additional speed optimizations
        "use_grayscale": False,   # Disable grayscale when using GPU for better quality
        "resize_input": False,    # Disable input resizing when using GPU
        "input_size": 640         # Use full size when GPU is available
    }

    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Config(**args)
    predictor = Predictor(config, verbose=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Optimize camera settings for real-time processing
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   # Even lower resolution for faster processing
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Even lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)            # Set consistent FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer to minimize lag

    # Additional camera optimizations
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure for consistent lighting
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus for consistent performance

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {frame_width}x{frame_height}")

    # Optional: Video saving setup (comment out if not needed for better performance)
    save_video = True  # Set to True if you want to save the output
    if save_video:
        # os.makedirs("output", exist_ok=True)
        # output_path = "output/realtime_output.avi"
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fps = 30.0  
        # out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # Extract base name of input video (without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("output", f"{video_name}_processed.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Use original FPS if available
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"💾 Saving processed video to: {output_path}")

    else:
        out_writer = None

    # Set up display window
    display_width = 640
    display_height = 480

    cv2.namedWindow('Real-Time Camera Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-Time Camera Detection', display_width, display_height)

    # Real-time optimization variables
    frame_skip = 5  # Process every 3rd frame for real-time performance
    frame_count = 0
    last_processed_frame = None

    print("Starting real-time camera processing...")
    print("Press 'q' to quit, 's' to toggle video saving, 'g' to toggle grayscale")
    print(f"Processing every {frame_skip}th frame for optimal performance")
    print(f"Device: {config.device}")
    print(f"Optimizations: Grayscale={config.use_grayscale}, Input resize={config.resize_input}")

    # GPU memory monitoring (if using GPU)
    if config.device.startswith('cuda'):
        print(f"GPU Memory before start: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        # Clear GPU cache for optimal performance
        torch.cuda.empty_cache()

    import time
    fps_counter = 0
    fps_start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        frame_count += 1
        fps_counter += 1

        # Skip frames for real-time performance
        if frame_count % frame_skip == 0:
            try:
                # Process frame with AI
                _, processed_frame = predictor.recognize(frame)
                if processed_frame is not None:
                    last_processed_frame = processed_frame
                    if save_video and out_writer is not None:
                        out_writer.write(processed_frame)
            except Exception as e:
                print(f"Processing error: {e}")
                processed_frame = frame
        else:
            # Use last processed frame or original frame for smooth display
            if last_processed_frame is not None:
                processed_frame = last_processed_frame
                if save_video and out_writer is not None:
                    out_writer.write(last_processed_frame)
            else:
                processed_frame = frame

        # Display frame
        if processed_frame is not None:
            cv2.imshow('Real-Time Camera Detection', processed_frame)

        # Calculate and display FPS every second
        if fps_counter >= 30:  # Every 30 frames
            elapsed_time = time.time() - fps_start_time
            current_fps = fps_counter / elapsed_time

            # Display performance info
            if config.device.startswith('cuda'):
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"Display FPS: {current_fps:.1f} | GPU Memory: {gpu_memory:.1f} MB")
            else:
                print(f"Display FPS: {current_fps:.1f} | Device: CPU")

            fps_counter = 0
            fps_start_time = time.time()

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_video = not save_video
            print(f"Video saving: {'ON' if save_video else 'OFF'}")
            if save_video and out_writer is None:
                os.makedirs("output", exist_ok=True)
                output_path = "output/realtime_output.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
            elif not save_video and out_writer is not None:
                out_writer.release()
                out_writer = None
        elif key == ord('g'):
            predictor.use_grayscale = not predictor.use_grayscale
            print(f"Grayscale mode: {'ON' if predictor.use_grayscale else 'OFF'}")
        elif key == ord('r'):
            predictor.resize_input = not predictor.resize_input
            print(f"Input resize: {'ON' if predictor.resize_input else 'OFF'}")

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    print("Camera processing stopped.")
