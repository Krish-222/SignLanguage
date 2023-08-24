import cv2

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Set the ROI dimensions
roi_x = 100
roi_y = 100
roi_width = 200
roi_height = 200


while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame=cv2.flip(frame,1)

    # Draw a rectangle to mark the ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    cv2.imshow('Camera Feed', frame)  # Display the frame with ROI

    # Capture the ROI
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale ROI
    gaussblur = cv2.GaussianBlur(gray_roi, (5, 5), 2)

    # Apply adaptive thresholding to create a binary image
    final_image = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2.8)

    # Resize the cbinary ROI to 128x128
    resized_binary_roi = cv2.resize(final_image, (200, 200))

    # Display the resized binary ROI
    cv2.imshow('Resized Binary ROI', resized_binary_roi)
    if(cv2.waitKey(1) & 0XFF == ord('c')):

    # Save the resized binary ROI to a file
      cv2.imwrite('newImage3.png', resized_binary_roi)
      break

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
