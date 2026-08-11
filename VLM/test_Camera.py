import cv2
print("Attempting to open camera...", flush=True)
cap = cv2.VideoCapture(4) # or cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Failed to open camera.", flush=True)
else:
    print("Camera opened successfully! Reading one frame...", flush=True)
    ret, frame = cap.read()
    if ret:
        print("Frame read successfully! Visual window should open.", flush=True)
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(2000) # Keep open for 2 seconds
    else:
        print("Failed to read frame.", flush=True)
    cap.release()
cv2.destroyAllWindows()