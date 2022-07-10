from scipy.spatial.distance import cosine
import hashlib
import numpy as np
from utils import utils


class FaceSimilarity:
    """Check of the similarity between two faces
    Model gives similarity score between 2 images, Lower the score more the similarity
    greater the score less the similairty 

    1.  This class have all the necessary cache things. 
        1. Cache the cropped face
        2. Cache the similarrity between 2 faces

    Score:
        (0 <= score >= 1)

        0: Means Full similar
        1: Not Similar
    """

    def __init__(self):
        """INIT contains theshold -> THRESH
        lower the score given by model more the similar and vice-versa

        Change the threshold according to your logic.
        """
        self.THRESH = 0.5

    def get_similarity(self, images: list([np.array, np.array])) -> list([int, bool]):
        """Get the face similarity between 2 selfies or human image.

        Args:
            images (list): [image1, image2] two images to get similarity off

        Returns:
            list : 
                int : Face similarity scrore
                bool: Boolean flag True if faces are similar and False if not similar
        """

        # Getting the image hashes
        score, match = 0.0, True
        image_hash_dict = {hashlib.md5(
            i.tobytes()).hexdigest(): i for i in images}
        image_hashes = list(image_hash_dict.keys())

        if len(image_hash_dict) > 1:

            # Checking if this image already in similarity cache
            for image_hash in image_hashes:
                if (image_hash in utils.similarity_cache) and (search in utils.similarity_cache[image_hash]):
                    search = set(images) - set([image_hash])
                    _, score = utils.similarity_cache[image_hash].index(search)
            # If caches result not found
            embeddings = utils.get_embeddings(image_hash_dict)
            # Comparing embedings
            score = cosine(embeddings[0], embeddings[1])
            match = score <= self.THRESH
            # If images are similar save in cache
            if match:
                utils.similarity_cache[image_hashes[0]] = [
                    image_hashes[1], score]

        return [score, match]


face_similarity = FaceSimilarity()
