import cv2
import numpy as np
class MediaProcessor:
    def __init__(self, detector, img_processor):
        self.detector = detector
        self.imgproc = img_processor

    def process(self, source):
        if source.lower().endswith((".jpg", ".jpeg", ".png")):
            self._process_image(source)
        elif source.lower().endswith((".mp4", ".avi", ".mkv")):
            self._process_video(source)
        else:
            raise ValueError("Unsupported file type!")

    # -------------------- IMAGE HANDLING --------------------
    def _process_image(self, path):
        img = cv2.imread(path)
        boxes = self.detector.detect(img)

        annotated, crops = self.imgproc.annotate_and_crop(
            img,
            boxes,
            self.detector.run_ocr
        )

        # Resize for display
        annotated = self.imgproc.resize_for_screen(annotated)

        # If crop exists
        crop_window = None
        if len(crops) > 0:
            crop, text = crops[0]
            crop = self.imgproc.resize_for_screen(crop)
            cv2.putText(crop, text, (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            crop_window = crop

        # Build one display frame (side by side WITHOUT concatenation)
        display = self._make_side_by_side_canvas(annotated, crop_window)

        cv2.imshow("License Plate Result", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Canvas Layout
    def _make_side_by_side_canvas(self, left_img, right_img):
        h = max(left_img.shape[0], right_img.shape[0] if right_img is not None else 0)
        w = left_img.shape[1] + (right_img.shape[1] if right_img is not None else 0)

        canvas = 255 * np.ones((h, w, 3), dtype=np.uint8)

        # place left side
        canvas[0:left_img.shape[0], 0:left_img.shape[1]] = left_img

        # place right side
        if right_img is not None:
            canvas[0:right_img.shape[0], left_img.shape[1]:left_img.shape[1]+right_img.shape[1]] = right_img

        return canvas

    # -------------------- VIDEO HANDLING --------------------
    def _process_video(self, path):
        cap = cv2.VideoCapture(path)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes = self.detector.detect(frame)

            annotated, _ = self.imgproc.annotate_and_crop(
                frame,
                boxes,
                self.detector.run_ocr
            )

            annotated = self.imgproc.resize_for_screen(annotated)
            cv2.imshow("License Plate Video", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
