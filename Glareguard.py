import cv2
import numpy as np

# Path to the video file
video_file_path = r'C:\Users\Yash\OneDrive\Desktop\r2.mp4'
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("Error: Unable to access the video file.")
    exit()

# FPS for playback speed
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_time = int(1000 / fps)

# Brightness levels for the bar display
brightness_levels = {
    'high': 1.0,
    'medium': 0.6,
    'low': 0.3,
    'very_low': 0.1
}

# Parameters for the brightness bar display
bar_height = 300
bar_width = 50
bar_gap = 40
bar_color = (0, 255, 255)
background_color = (50, 50, 50)

# Detection parameters
distance_threshold = 500
scale_factor = 0.05
min_aspect_ratio, max_aspect_ratio = 1.5, 4.0
min_area = 50  # Increased min area for filtering out small lights

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame.")
        break

    # Initialize brightness states for each headlight
    left_brightness = 'high'
    right_brightness = 'high'

    # Grayscale and threshold for bright areas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour to detect headlights
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue

        # Filter contours by position to detect lights closer to the bottom half
        if y < frame.shape[0] // 2:
            continue

        # Determine the side of the frame
        if x + w // 2 < frame.shape[1] // 2:
            side = 'left'
        else:
            side = 'right'

        # Calculate distance to the frame center
        contour_center = (x + w // 2, y + h // 2)
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        distance_pixels = int(np.linalg.norm(np.array(contour_center) - np.array(frame_center)))
        distance_meters = distance_pixels * scale_factor

        # Adjust brightness based on distance and side
        if distance_pixels < distance_threshold:
            if distance_pixels < 100:
                brightness = 'very_low'
            elif distance_pixels < 250:
                brightness = 'low'
            else:
                brightness = 'medium'
        else:
            brightness = 'high'

        if side == 'left':
            left_brightness = brightness
        else:
            right_brightness = brightness

        # Draw rectangle and distance text for detected headlights
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Distance: {distance_meters:.2f} m', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Create a separate window for brightness bars
    bar_window = np.full((bar_height + 20, 2 * bar_width + bar_gap, 3), background_color, dtype=np.uint8)

    # Set heights based on brightness levels
    left_bar_height = int(bar_height * brightness_levels[left_brightness])
    right_bar_height = int(bar_height * brightness_levels[right_brightness])

    # Draw the left and right brightness bars
    cv2.rectangle(bar_window, (10, bar_height - left_bar_height), (10 + bar_width, bar_height), bar_color, -1)
    cv2.rectangle(bar_window, (10 + bar_width + bar_gap, bar_height - right_bar_height), 
                  (10 + 2 * bar_width + bar_gap, bar_height), bar_color, -1)

    # Draw outlines for clarity
    cv2.rectangle(bar_window, (10, 0), (10 + bar_width, bar_height), (255, 255, 255), 2)
    cv2.rectangle(bar_window, (10 + bar_width + bar_gap, 0), (10 + 2 * bar_width + bar_gap, bar_height), (255, 255, 255), 2)

    # Add labels for clarity
    cv2.putText(bar_window, 'Left Headlight', (10, bar_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bar_window, 'Right Headlight', (10 + bar_width + bar_gap, bar_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show both windows
    cv2.imshow("Oncoming Vehicle Headlight Detection", frame)
    cv2.imshow("Brightness Levels", bar_window)

    # Normal playback speed
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()