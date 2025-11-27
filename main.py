from models.license_plate_detector import LicensePlateDetector
from utils.media_processor import MediaProcessor
from utils.image_utils import ImageProcessor

def main():
    # YOLO model path
    YOLO_MODEL = "weights\plate_detector.pt"  
    SOURCE = r"D:\video1.mp4"  

    detector = LicensePlateDetector(YOLO_MODEL)
    image_processor = ImageProcessor()
    media = MediaProcessor(detector)

    media.process(SOURCE)

if __name__ == "__main__":
    main()
