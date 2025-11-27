from models.license_plate_detector import LicensePlateDetector
from utils.media_processor import MediaProcessor
from utils.image_utils import ImageProcessor


def main():
    YOLO_MODEL = "weights/plate_detector.pt"
    SOURCE = r"C:\Users\LENOVO\Downloads\img\123.jpg" 

    detector = LicensePlateDetector(YOLO_MODEL)
    processor = ImageProcessor()
    media = MediaProcessor(detector, processor)

    media.process(SOURCE)


if __name__ == "__main__":
    main()
