
from ultralytics import YOLO

#used absolute path to make it work, using a relative path gives a FileNotFound error for some reason
path = '/Users/shauryag./Documents/GitHub/Project3/data/data.yaml'

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=path, epochs=1, batch = 16, imgsz = 900, name = 'pcb identifier')

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('data/evaluation/ardmega.jpg')

results = model(['data/evaluation/ardmega.jpg', 'data/evaluation/arduno.jpg', 'data/evaluation/rasppi.jpg'])  # return a list of Results objects

model.predict('data/evaluation/ardmega.jpg', save=True, imgsz=900, conf=0.5)
model.predict('data/evaluation/arduno.jpg', save=True, imgsz=900, conf=0.5)
model.predict('data/evaluation/rasppi.jpg', save=True, imgsz=900, conf=0.5)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Export the model to ONNX format
success = model.export(format='onnx')