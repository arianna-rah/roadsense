# RoadSense: AI-Powered Road Condition Classifier

RoadSense is an intelligent road condition analysis system that uses deep learning models to classify road surfaces and detect anomalies in real-time. The application provides a web interface for uploading road images and receiving instant analysis results, helping drivers and infrastructure managers assess road safety conditions.

## Features

- **Road Condition Classification**: Classifies road images into five categories: dry, wet, standing water, snow, and ice
- **Anomaly Detection**: Uses YOLOv11 for detecting road anomalies and hazards
- **Real-time Analysis**: Fast inference with optimized models for quick results
- **Web Interface**: User-friendly Next.js frontend for easy image upload and result visualization
- **RESTful API**: FastAPI backend providing endpoints for classification and detection
- **Docker Support**: Containerized backend for easy deployment

## Tech Stack

### Backend
- **Python 3.11**
- **FastAPI**: High-performance web framework
- **PyTorch & TorchVision**: Deep learning framework
- **Ultralytics YOLOv11**: Object detection model for damage detection
- **MobileNet-V3**: Custom-trained CNN for condition analysis
- **Docker**: Containerization

### Frontend
- **Next.js 14**: React framework
- **Tailwind CSS**: Utility-first CSS framework

### Development
- **Model Training**: Trained MobileNet-V3 Large CNN model for classification of road conditions
- **Data Processing**: Image preprocessing and augmentation (cropping, etc.)

## Project Structure

```
road-classifier/
├── backend/                 # FastAPI backend application
│   ├── main.py             # Main API server with endpoints
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Backend containerization
│   ├── best_model (1).pth  # Trained classification model
│   └── best (2).pt         # YOLO detection model
├── model-dev/              # Model development and training
│   ├── classifier-train.py # Training script
│   └── classifier-test.py  # Testing script
├── roadsense/              # Next.js frontend application
│   ├── app/                # Next.js app directory
│   ├── package.json        # Node.js dependencies
│   └── next.config.ts      # Next.js configuration
└── README.md               # Project documentation
```

## Installation

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11 (for backend development)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t road-backend .
   docker run -p 8000:8000 road-backend
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd roadsense
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:3000`

## Usage

1. Start both backend and frontend services as described above
2. Open your browser and navigate to `http://localhost:3000`
3. Upload a road image using the upload interface
4. View the classification results and any detected anomalies

## API Documentation

### POST /predict
Classifies the road condition in an uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "predicted": "dry",
  "confidence": 0.95,
  "probabilities_per_class": {
    "dry": 0.95,
    "wet": 0.03,
    "standing_water": 0.01,
    "snow": 0.005,
    "ice": 0.005
  }
}
```

### POST /yolo-predict
Detects anomalies in an uploaded road image using YOLO.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "num_anomalies": 2,
  "result_img": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

## Model Training

Training of the models was done in Kaggle Notebooks. Code run for training and testing of the CNN (trained the head/classifier of MobileNet-V3 Large, froze three of the base layers) can be found in the model-dev directory; the dataset used for training was the RSCD dataset. While training, hyper-parameters were fine-tuned, and early stopping was implemented to prevent overfitting. For the YOLO model, training was done on the RDD-2022 dataset

Model Metrics (CNN/MobileNet):
- Accuracy: 95.03%
- F1 Score: 0.9504159688949585
- Confusion Matrix: 
     tensor([[3740,  223,   19,    6,   12],
             [ 130, 3645,  217,    4,    4],
             [  21,  238, 3729,   11,    1],
             [   7,    5,    9, 3930,   49],
             [   2,    2,    1,   33, 3962]])

Model Metrics (YOLOv11n)
- Precision: 0.609
- Recall: 0.532      
- mAP50: 0.565      
- mAP50-95: 0.304

## Acknowledgments

- Built with state-of-the-art computer vision models
- Inspired by the need for safer road infrastructure monitoring
- Thanks to the open-source community for the amazing tools and libraries
