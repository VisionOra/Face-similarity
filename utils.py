import time
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from model import VGGFace
from keras_vggface.utils import preprocess_input
from face_detection.detect_face import inference as detect_face


class Utils:
    """Supporting functions to calculate the face similarity
    """

    def __init__(self):
        """Initilizer

        1. model: Face similarity model with segnet
        2. detector: Face detection MTCNN model
        3. cache_crop_face: Cache Cropped faces according to the images hash
        4. similarity_cache: Cache the similarity of two images

        """
        # Face similarity model
        self.model = VGGFace(model='senet50', include_top=False,
                             input_shape=(224, 224, 3), pooling='avg')
        # Face detector model
        self.detector = MTCNN()
        # cache crop faces
        self.cache_crop_face = dict()
        # cache similarity of two images
        self.similarity_cache = dict()

    def extract_face(self, pixels: np.asarray, required_size: tuple = (224, 224)) -> np.asarray:
        """Extract faces from a full size image, This function use
        MTCNN to crop faces from the full-images 

        Args:
            pixels (np.asarray): Image to get image from
            required_size (tuple, optional): _description_. Defaults to (224, 224).

        Returns:
            np.asarray: Detected Face image 
        """

        # detect faces in the image
        results = self.detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        image = cv2.resize(face, dsize=required_size,
                           interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return image

    # extract faces and calculate face embeddings for a list of photo files
    def get_embeddings(self, image_hash_dict: dict, required_size=(224, 224)) -> np.asarray:
        """Return the embeddings of two faces, to find the cosine similarity
        betweenn them.

        Args:
            image_hash_dict (dict): hash and np.array of images
                ```
                {
                    hash_of_image: np.asarray(image)
                }
                ```
        Returns:
            np.asarray: Embeddings of face
        """
        image_hashes = list(image_hash_dict.keys())

        # Caching cropped faces
        for img_hash, image in image_hash_dict.items():
            if img_hash not in self.cache_crop_face:
                face = detect_face(image)
                self.cache_crop_face[img_hash] = cv2.resize(face, dsize=required_size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
            
                

        # extract faces
        faces = [self.cache_crop_face[image_hashes[0]],
                 self.cache_crop_face[image_hashes[1]]]
        # convert into an array of samples
        st_time = time.time()
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(faces, version=2)
        # perform prediction
        yhat = self.model.predict(samples)
        return yhat


utils = Utils()
