
# Face Similarity
Get the face Similarity between 2 images

![System Flow](diagrams/face_similarity.drawio.png)

## Installation

### Technology
1. Python=3.8

```bash
Make Python virtual environment (you can use of your own choice)
# Make conda environment
conda create -n FCR python=3.8
# Intall Requirements
pip install -r requirements.txt
## Run Code Test face similarity
python main.py
## Run code test face-detection
python face_detection_main.py 
```

### Cmake Installation guide
[Cmake Installation Guide](https://cmake.org/install/) 

### Install conda guide
[Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html )

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



