from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Load a single image
img = cv2.imread('potholeimage3.jpeg')
if img is None:
    print("Error: Could not open or find the image.")
    exit()

# Resize the image
img = cv2.resize(img, (1020, 500))
h, w, _ = img.shape

# Predict using the model
results = model.predict(img)

# Iterate over the results
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image with detections
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
