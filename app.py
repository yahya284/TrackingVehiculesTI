import cv2
from ultralytics import YOLO, solutions

# Load YOLO model
model = YOLO("yolo11n.pt")
names = model.model.names

# Open video
cap = cv2.VideoCapture("test_videos/vid.mp4")
if not cap.isOpened():
    print("Error reading video file. Check the file path.")
    exit()

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Speed multiplier
speed_multiplier = 3  # Adjust this to make video faster (e.g., 2 for 2x speed, 3 for 3x speed)
new_fps = int(fps * speed_multiplier)

# Create VideoWriter with adjusted FPS
video_writer = cv2.VideoWriter("speed.mp4", cv2.VideoWriter_fourcc(*"mp4v"), new_fps, (w, h))

# Fix line_pts to fit within the video resolution
line_pts = [(100, 300), (1800,300)]  # Ensure line points are within video width

# SpeedEstimator setup
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=False,
)

print("Frame resolution: ", w, "x", h)
print("Original FPS:", fps, "New FPS:", new_fps)

while cap.isOpened():
    success, im0 = cap.read()
    if not success or im0 is None or im0.size == 0:
        print("Frame is empty or invalid. Stopping video processing.")
        break

    # Track objects in the frame
    tracks = model.track(im0, persist=True)
    if not tracks:
        print("No objects tracked in this frame.")
        continue

    # Estimate speed and check validity
    im0 = speed_obj.estimate_speed(im0, tracks)
    if im0 is None or im0.size == 0:
        print("Processed frame is empty or invalid. Skipping.")
        break

    # Write the frame to the output video
    video_writer.write(im0)

    # Display the video frame
    cv2.imshow("Processed Video", im0)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
