# import cv2

# # Open the camera (0 = /dev/video0)
# cap = cv2.VideoCapture(0)

# # Optional: set resolution (adjust if needed)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open camera")
#     exit()

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Error: Can't receive frame")
#         break

#     # Show the frame
#     cv2.imshow("Camera Feed", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
print("Opened:", cap.isOpened())
cap.release()