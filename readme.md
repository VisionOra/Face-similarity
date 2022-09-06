# Face Match Application
In this article we are going to have some insights that how AI is helping us in matching faces
like most of our smart phones do for security purposes. 

Github Link : https://github.com/SohaibAnwaar/Face-similarity

### Steps to Perform FaceMatch
1. Detect Face from the images
2. Give cropped faces to the model to get features
3. Get the cosine similarity between the faces.


## Diagram 
[Sohaib Anwaar](https://www.linkedin.com/in/sohaib-anwaar-4b7ba1187/) introduced some more architecture level flows so that we can make our system more effecient, Below you can see the diagram

### Steps performed in Architecture
1. Pass images to Face recogination model
2. See if we already have the embeddings of the face available or not. 
3. If embeddings already available get embeddings from the database
4. Else give faces to the model and get the embeddings.
5. Get the cosine smilarity of the embeddings 
6. If cosine similarity > 0.5 than match else not match

![System Flow](diagrams/face_similarity.drawio.png)

## Lets get started with the code part

### Technology
1. Python=3.8

```bash
Make Python virtual environment (you can use of your own choice)
# Make conda environment
conda create -n FCR python=3.8
# Clone the github
Git clone https://github.com/SohaibAnwaar/Face-similarity.git
# Intall Requirements
pip install -r requirements.txt
## Run Code Test face similarity
python main.py
## Run code test face-detection
python face_detection_main.py 
```

## Third Party installations if you got errors.
### Cmake Installation guide

1. [Cmake Installation Guide](https://cmake.org/install/) 

### Conda install  guide
1. [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html )

## Lets dive into the code part

Directory Structure
```
├── Debug_face_similarity.ipynb       # For Jupyter notebook testing
├── diagrams # Assets of the git (pictures used to represent things in readme file)
│   ├── face_recogination.jpg
│   ├── face_similarity.drawio
│   └── face_similarity.drawio.png
├── face_detection                    # Face detection folder
│   ├── box_utils_numpy.py            # Face detection utils code
│   ├── Debug_face_detection.ipynb    # Jupyter notebook to test Face Detection code
│   ├── detect_face.py                # Face detection python code so that we can use it as module
│   ├── model # Model Weights
│   │   ├── version-RFB
│   │   │   ├── RFB-320.mnn
│   │   │   ├── RFB-320-quant-ADMM-32.mnn
│   │   │   └── RFB-320-quant-KL-5792.mnn
│   │   └── version-slim
│   │       ├── slim-320.mnn
│   │       └── slim-320-quant-ADMM-50.mnn
│   └── readme.md # Read me file
├── face_detection_main.py            # Face detection Main file
├── face_similarity.py                # Face Similarity file
├── main.py                           # Main file to detect face and than perform face similarity
├── model.py                          # we can get model prediction from here
├── model_weights
│   └── read.md
├── readme.md
├── requirements.txt                  # Requirements to run this python code
├── testing_images
│   ├── test2.jpeg
│   ├── test3.jpeg
│   ├── test4.jpg
│   ├── test5.jpg
│   └── test.jpg
└── utils.py
```

## Read Image
Reading image with the help of matplotlib
```python
image1 = "test.jpg"
image2 = "test2.jpeg"
image1 = plt.imread(image1)
image2 = plt.imread(image2)
```
## MD5 of Image For Cache
we make the md5 of image and store it in our dictinary (for long run you can replace it with your database) to use it as a cache, for-example if someone pass the image1 again and again, so we dont need to run the face detection again and again on the image we just compare the MD5 hashed of the image and get the features of it. 

Acutally, we saved model inference time here. 

```python
image_hash_dict = {hashlib.md5(i.tobytes()).hexdigest(): i for i in images}
```

## Running face recoginion model

We are using the lightest face recogination model here, even you can also use it in your mobile which is ``MTCNN``

Detect Face
```python
def inference(image_ori: np.asarray) -> np.asarray:
    """Crop face from Image
    1. If their are no face in image it will return 0
    2. If their are more than 1 face it will return 0
    
    Args:
        image_ori = Full image with face
    
    Output:
        np.asarray: Cropped face image numpy array
    
    """
    # Variables
    input_size = (320,240)
    threshold = 0.7
    # Loading model
    interpreter = MNN.Interpreter(model_path)
    priors = define_img_size(input_size)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # Preprocessing Image
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(input_size))
    image = (image - image_mean) / image_std
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    # Running Prediction
    interpreter.runSession(session)
    scores = interpreter.getSessionOutput(session, "scores").getData()
    boxes = interpreter.getSessionOutput(session, "boxes").getData()
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = box_utils.center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, threshold)
    # cropping face
    if len(boxes) == 1:
        y1, x1, y2, x2 = boxes[0]
        cropped_face = image_ori[x1: x2, y1: y2]
        return cropped_face
    else:
        raise Exception("More than 1 faces in picture Or no Face In Image")
```

## Checking the face-similarity
Now as we are done with the face detection and other preprocessing things lets get the similarity between 2 faces.
```python
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

```
Yayyy All done:D Hope you guys like my article, You guys can find my whole codebase on github mentioned at top of the article.


## Support me, Follow me, Stay Connected with me for more awesome articles.

* Sohaib Anwaar
* gmail          : sohaibanwaar36@gmail.com
* linkedin       : [Have Some Professional Talk here](https://www.linkedin.com/in/sohaib-anwaar-4b7ba1187/)
* Stack Overflow : [Get my help Here](https://stackoverflow.com/users/7959545/sohaib-anwaar)
* Kaggle         : [View my master-pieces here](https://www.kaggle.com/sohaibanwaar1203)


## Timing of Model Inference
# Face Similarity Results and Time

```
------------------------------------------------------
Score 0.5789089202880859, Match False
Expected Results [Different] - Different Images
End Time 2.1229381561279297
-----------------------------------------------------------



------------------------------------------------------
Score 0.0, Match True
Expected Results [Same] - Same Images
End Time 0.007751941680908203
-----------------------------------------------------------
priors nums:4420



------------------------------------------------------
Score 0.7787326276302338, Match False
Expected Results [Different] - One Image is alreaedy in System Other one is new (Testing cache)
End Time 0.22133398056030273
-----------------------------------------------------------



------------------------------------------------------
Score 0.7787326276302338, Match False
Expected Results [Different] - Both of the iamges are available in system alreayd
End Time 0.1980288028717041
-----------------------------------------------------------
```

# Face Detection Results and Time

```
priors nums:4420
Inference Time 0.030576705932617188
priors nums:4420
Inference Time 0.020635128021240234
priors nums:4420
Inference Time 0.020209312438964844
priors nums:4420
Inference Time 0.02044391632080078
priors nums:4420
Inference Time 0.02028799057006836
priors nums:4420
Inference Time 0.02002716064453125
priors nums:4420
Inference Time 0.020220041275024414
priors nums:4420
Inference Time 0.020006179809570312
priors nums:4420
Inference Time 0.019464969635009766
```



