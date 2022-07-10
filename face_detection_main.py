from face_detection.detect_face import inference as detect_face
import time
import cv2


def test():
    """Detect Face detection model
    """
    for i in range(1, 10):
        st = time.time()
        img_path = "testing_images/test2.jpeg"
        image_ori = cv2.imread(img_path)
        face = detect_face(image_ori)
        print(f"Inference Time {time.time() - st}")


test()