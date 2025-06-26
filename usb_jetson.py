import os
import shutil
import time
import threading
import hashlib
import cv2

class USBAdsManagerJetson:
    def __init__(self, current_ads_folder="current_ads", usb_ads_subfolder="ads"):
        self.current_ads_folder = current_ads_folder
        self.usb_ads_subfolder = usb_ads_subfolder
        self.processed_usbs = set()
        self.display_thread = None
        self.stop_display = False
        self.media_root = "/media/jetson/"

        os.makedirs(self.current_ads_folder, exist_ok=True)
        self.start_ad_display()

    def find_usb_drives(self):
        usb_drives = []
        if os.path.exists(self.media_root):
            for usb_name in os.listdir(self.media_root):
                usb_path = os.path.join(self.media_root, usb_name)
                if os.path.isdir(usb_path) and os.access(usb_path, os.R_OK):
                    usb_drives.append(usb_path)
        return usb_drives

    def find_ads_in_usb(self, usb_path):
        ads_folder = os.path.join(usb_path, self.usb_ads_subfolder)
        if not os.path.exists(ads_folder):
            return []
        return [
            os.path.join(ads_folder, f)
            for f in os.listdir(ads_folder)
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')) and os.path.isfile(os.path.join(ads_folder, f))
        ]

    def copy_ads_from_usb(self, usb_path):
        print(f"USB detected: {usb_path}")
        video_files = self.find_ads_in_usb(usb_path)
        if not video_files:
            print("No ads found in USB.")
            return

        for source_file in video_files:
            try:
                file_name = os.path.basename(source_file)
                dest_path = os.path.join(self.current_ads_folder, file_name)

                # Avoid name conflict
                counter = 1
                base, ext = os.path.splitext(file_name)
                while os.path.exists(dest_path):
                    dest_path = os.path.join(self.current_ads_folder, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy2(source_file, dest_path)
                print(f"Copied: {file_name}")
            except Exception as e:
                print(f"Error copying file: {e}")

    def get_video_files(self):
        return [
            os.path.join(self.current_ads_folder, f)
            for f in os.listdir(self.current_ads_folder)
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        ]

    def display_ads_loop(self):
        print("Starting ad display...")
        while not self.stop_display:
            files = self.get_video_files()
            if not files:
                time.sleep(5)
                continue

            for video in files:
                if self.stop_display:
                    break
                print(f"Playing: {os.path.basename(video)}")
                cap = cv2.VideoCapture(video)
                if not cap.isOpened():
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                delay = int(1000 / fps)
                while not self.stop_display:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.namedWindow('Ad Display', cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty('Ad Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Ad Display", frame)
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        self.stop_display = True
                        break
                cap.release()
        cv2.destroyAllWindows()
        print("Ad display stopped.")

    def start_ad_display(self):
        self.display_thread = threading.Thread(target=self.display_ads_loop, daemon=True)
        self.display_thread.start()

    def monitor_usb_drives(self):
        print("Monitoring USB drives...")
        last_drives = set()
        while True:
            try:
                current_drives = set(self.find_usb_drives())
                new_drives = current_drives - last_drives
                for drive in new_drives:
                    self.copy_ads_from_usb(drive)
                last_drives = current_drives
                time.sleep(2)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"USB monitor error: {e}")
                time.sleep(5)
        self.stop_display = True

def main():
    print("Jetson Ad Display System")
    manager = USBAdsManagerJetson()
    manager.monitor_usb_drives()

if __name__ == "__main__":
    main()
