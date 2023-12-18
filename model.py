
from ultralytics import YOLO

#used absolute path to make it work, using a relative path gives a FileNotFound error for some reason
#path = '/Users/shauryag./Documents/GitHub/Project3/data/data.yaml'
path = 'data/data.yaml'



# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=path, epochs=10, batch = 16, imgsz = 900)
