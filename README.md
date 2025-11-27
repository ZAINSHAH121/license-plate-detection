# 🚗 License Plate Detection & OCR System  
### YOLO + TrOCR | Image & Video Plate Recognition | Side-by-Side Output

This project detects vehicle license plates using **YOLOv8**, performs OCR using **Microsoft TrOCR**, and displays results with a clean interface:

- **Left Side →** Original Image + YOLO Annotations + OCR  
- **Right Side →** Cropped Plate Images + OCR Text  
- **Video Mode →** YOLO + OCR (no side-by-side for performance)

---

## ✨ Features

✔️ YOLOv8-based license plate detection  
✔️ OCR with Microsoft TrOCR  
✔️ Side-by-side visualization for images  
✔️ Real-time video processing  
✔️ Automatic OCR text cleaning (regex)  
✔️ Modular, clean architecture (detector + processor + utils)  
✔️ Easy to extend and customize  

---

## 📂 Project Structure
License-plate-detection/
│
├── main.py
│
├── models/
│ └── license_plate_detector.py
│
├── utils/
│ ├── media_processor.py
│ └── image_processor.py
│
├── weights/
│ └── plate_detector.pt
│
├── requirements.txt
└── README.md


---

## 📦 Installation

### 1️⃣ Create & Activate Virtual Environment
```bash
conda create -n yoloenv python=3.10 -y
conda activate yoloenv

##2️⃣ Install Dependencies
pip install -r requirements.txt

⚙️ Running the Project
▶️ Run Detection on Image or Video
python main.py


## How OCR Works

This project uses Microsoft TrOCR, a Transformer-based OCR model.

OCR pipeline:

Extract plate crop from YOLO

Send crop → TrOCR

Clean text using regex:

text = re.sub(r"[^A-Z0-9\- ]", "", text).upper()


Display clean plate number

🚀 Customization
Change YOLO image size
results = self.model(img, imgsz=640)

Change visualization max window size
max_width = 1600
max_height = 1000

Control cropping output
side_by_side = cv2.hconcat([annotated] + resized_crops)

📌 Requirements Summary

Python 3.10+

PyTorch

Ultralytics YOLO

OpenCV

Transformers (TrOCR)

Pillow

Full list in requirements.txt

🤝 Contributing

Pull requests, improvements, and optimizations are welcome!

📜 License

This project is intended for learning and research purposes only.

👤 Author

Syed Zain Qaiser
Machine Learning Engineer | AI & Computer Vision Enthusiast
