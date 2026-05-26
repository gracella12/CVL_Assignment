import cv2
import pandas as pd
import  numpy as np

cap = cv2.VideoCapture("Walking.mp4")  # atau 0 untuk webcam

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS video: {fps}")

skip_to_second = 3
target_frame = int(fps * skip_to_second)

cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

ret, first_frame = cap.read()

# pilih objek yang ingin ditrack
roi = cv2.selectROI("Pilih Objek", first_frame, False, False)
cv2.destroyAllWindows()

x, y, w, h = roi
print(f"ROI dipilih: x={x}, y={y}, w={w}, h={h}")

template = first_frame[y:y+h, x:x+w]

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

print(f"Ukuran template: {template_gray.shape}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(frame, f"conf: {max_val:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Template Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()