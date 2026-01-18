from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from PIL import Image
from torchvision import transforms
import torch
import timm
import ultralytics
import io
import base64
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os

app = FastAPI()

origins = [
    "http://localhost:3000"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

class_to_idx = {0: "dry", 1: "wet", 2: "standing water", 3: "snow", 4: "ice"}

model = timm.create_model('tf_mobilenetv3_large_100.in1k', pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=5, 
                    bias=True))

state_dict_path = "best_model (1).pth"
state_dict = torch.load(state_dict_path, map_location=device)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

yolo_model = YOLO('best (2).pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try: 
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        w, h = img.size
        crop_box = (w/3, h/2, 2*w/3, h)
        cropped_img = img.crop(crop_box)
        cropped_img = img_transforms(cropped_img).unsqueeze(0).to(device)
        with torch.no_grad():
            label_pred = model(cropped_img)
            probs = torch.nn.functional.softmax(label_pred, dim=1)
            confidence, predicted = torch.max(probs, 1)
    
        probabilities_per_class = {"dry": float(probs[0][0].item()), 
                                "wet": float(probs[0][1].item()), 
                                "standing_water": float(probs[0][2].item()), 
                                "snow": float(probs[0][3].item()), 
                                "ice": float(probs[0][4].item()) }

        return {
            "predicted": class_to_idx[predicted.item()],
            "confidence": float(confidence.item()),
            "probabilities_per_class": probabilities_per_class
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/yolo-predict')
async def yolo_predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)) 

        results = yolo_model.predict(
        source=img,
        save=False,          
        conf=0.05,
        device = device           
        )
        num_anomalies = len(results[0].boxes)
        result_img = Image.fromarray(results[0].plot())
        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")

        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        result_url = f"data:image/jpeg;base64,{img_base64}"

        return {
            "num_anomalies": num_anomalies,
            "result_img": result_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  