import asyncio
import os
import cv2
import struct
import time

# Simulate an ad display system (displaying ads from folder)
async def display_ads(ad_queue, ad_folder="video", default_image_duration=30, fps=60):
    while True:
        # Get list of files in the 'ads' folder (only videos/images)
        ads = [ad for ad in os.listdir(ad_folder) if ad.endswith((".mp4", ".jpg", ".png", ".mov", ".MOV"))]
        
        if not ads:
            print("No ads found in the folder.")
            await asyncio.sleep(5)  # Retry after some time
            continue
        
        # Cycle through ads
        for ad in ads:
            print(f"Adding Ad to Queue: {ad}")
            await ad_queue.put(ad)  # Place the current ad ID in the queue
            
            # Handle displaying the ad
            ad_file_path = os.path.join(ad_folder, ad)
            if ad.endswith(".mp4") or ad.endswith(".mov") or ad.endswith(".MOV"):
                ad_cap = cv2.VideoCapture(ad_file_path)
                if not ad_cap.isOpened():
                    print(f"Error: Unable to open ad video: {ad}")
                    continue
                
                # Calculate video duration
                frame_count = int(ad_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = ad_cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / video_fps if video_fps > 0 else 30  # Fallback to 30 seconds
                
                print(f"Displaying video ad: {ad} for {duration:.2f} seconds")
                start_time = time.time()
                
#                while time.time() - start_time < duration:
                while True:
                    ret, ad_frame = ad_cap.read()
                    if not ret:
                        break  # Break out when video ends
                    cv2.namedWindow('Ad Display', cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty('Ad Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # Display the ad frame
                    cv2.imshow("Ad Display", ad_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    await asyncio.sleep(1 / video_fps)  # Control frame rate
            else:
                # For image ads, use default duration
                ad_image = cv2.imread(ad_file_path)
                if ad_image is None:
                    print(f"Error: Unable to load ad image: {ad}")
                    continue
                
                print(f"Displaying image ad: {ad} for {default_image_duration} seconds")
                start_time = time.time()
                
                while time.time() - start_time < default_image_duration:
                    cv2.imshow("Ad Display", ad_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    await asyncio.sleep(1 / fps)  # Control frame rate


# Stream frames to the server (street.mp4)
async def send_frames(server_ip='localhost', server_port=12345, fps=30, ad_queue=None):
    last_known_ad = "unknown"  # Cache the last valid ad ID

    while True:
        try:
            print(f"Trying to connect to {server_ip}:{server_port}...")
            reader, writer = await asyncio.open_connection(server_ip, server_port)
            print("Connected to the server")

            # Open video source (street.mp4)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Unable to open street.mp4.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No frame captured, exiting.")
                    break

                frame = cv2.resize(frame, (320, 240),interpolation=cv2.INTER_AREA) 

                # Get the current ad ID, fallback to the last known ad if queue is empty
                if not ad_queue.empty():
                    current_ad = await ad_queue.get()
                    last_known_ad = current_ad  # Update cache
                    ad_queue.task_done()
                else:
                    current_ad = last_known_ad  # Use cached value

                print(f"Using Ad ID: {current_ad}")

                # Compress the frame (JPEG encoding)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                data = buffer.tobytes()

                # Include the ad ID as part of the metadata
                ad_id_bytes = current_ad.encode() if current_ad else b"unknown"
                ad_id_size = struct.pack(">I", len(ad_id_bytes))
                frame_size = struct.pack(">I", len(data))

                try:
                    writer.write(ad_id_size + ad_id_bytes + frame_size + data)
                    await writer.drain()
                except (ConnectionResetError, BrokenPipeError):
                    print("Connection to server lost. Reconnecting...")
                    break  # Exit inner loop and reconnect

                # Control frame rate for the street.mp4 stream
                await asyncio.sleep(1 / fps)

        except Exception as e:
            print("Connection refused. Server might be down. Retrying in 5 seconds...")
            await asyncio.sleep(5)  # Wait before retrying
        except KeyboardInterrupt:
            print("Client interrupted by user. Exiting...")
            break
        finally:
            try:
                if 'cap' in locals():  # Check if cap was initialized
                    cap.release()
                if 'writer' in locals():
                    writer.close()
                    await writer.wait_closed()
            except Exception as e:
                print(f"Error during cleanup: {e}")

# Main function to run both tasks concurrently
async def main():
    ad_queue = asyncio.Queue()
    await asyncio.gather(
        display_ads(ad_queue),
        send_frames(server_ip='127.0.0.1', server_port=12345, ad_queue=ad_queue)
    )

# Run the client
asyncio.run(main())
