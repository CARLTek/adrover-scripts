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
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# Allowlist the required global
add_safe_globals([DetectionModel])

class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            out_im = detected_objects.plot()

        return detected_objects, out_im

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

    args = {
        "detector_weights": "models/yolov8x_person_face.pt",
        "checkpoint": "models/model_imdb_cross_person_4.22_99.46.pth.tar",
        "device": "cuda:0",
        "with_persons": True,
        "draw": True,
        "disable_faces": False
    }

    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Config(**args)
    predictor = Predictor(config, verbose=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Video saving setup
    os.makedirs("output", exist_ok=True)
    output_path = "output/processed_output.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0 or np.isnan(fps):
        fps = 25.0  # fallback

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0:
        print("Error: Invalid frame size.")
        cap.release()
        exit()

    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, processed_frame = predictor.recognize(frame)

        if processed_frame is not None:
            out_writer.write(processed_frame)

        cv2.imshow('Real-Time Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
