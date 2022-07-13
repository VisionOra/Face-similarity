from face_similarity import face_similarity
import matplotlib.pyplot as plt
import numpy as np
import time


def main(image1: np.array, image2: np.array):
    """Main function to check if code is working or not.

    1. At first this code will automatically download weights

    Args:
        image1 (np.array): Image one (Known Image)
        image2 (np.array): Image Two (Compare image with known)
    """
    # Reading images
    image1 = plt.imread(image1)
    image2 = plt.imread(image2)

    # Scores
    score, match = face_similarity.get_similarity([image1, image2])
    return score, match


image1 = "/Users/sohaibanwar/Documents/face_recogination/testing_images/test.jpg"
image2 = "/Users/sohaibanwar/Documents/face_recogination/testing_images/test2.jpeg"
image8 = "/Users/sohaibanwar/Documents/face_recogination/testing_images/test3.jpeg"



test_list = [
    [image1, image2],
    [image2, image2],
    [image8, image2],
    [image8, image2]
]

for (image1, image2) in test_list:
    st_time = time.time()
    score, match = main(image1, image2)
    print("\n\n\n------------------------------------------------------")
    print(f"Score {score}, Match {match}")
    print(f"End Time {time.time() - st_time}")
    print("-----------------------------------------------------------")
