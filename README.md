# vehicle-reidentification

Vehicle Reidentification

# Methods

1. HOG (Histogram of Oriented Gradients)
2. Histogram (Color Histogram)
3. DNN (Deep Neural Network)

# Installation

```
pip install -r requirements.txt
```

# Run

```
(cs5330) ➜  vehicle-reidentification git:(main) ✗ python main.py -h
usage: main.py [-h] -i INPUT -r REFERENCE_IMG [-f {hog,histogram,resnet,osnet,composite}]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input video
  -r REFERENCE_IMG, --reference_img REFERENCE_IMG
                        path to reference image
  -f {hog,histogram,resnet,osnet,composite}, --feature {hog,histogram,resnet,osnet,composite}
                        Feature type to use for comparison
```

# Evaluation
  
video name in brackets of Vehicle Type

| Vehicle Type | Histogram | HOG | SIFT | ResNet | OSNet |
| ------------ | --------- | --- | ------ | ---- | ----- |
| red car (front) | 70% | 70% | 70% | 80% | 80% |
| white car (front) | 20% | 20% | 70% | 80% (1 FP) | 80% |
| black car (back) | 80% | 80% | 70% | 95% | 95% |
| blue car (back) | 80% | 70% | 70% | 95% | 95% |
| black car (front2) | 100% (2 FP) | 95% | 80% (1 FP) | 100% | 100% |
| white car (front2) | 70% (2 FP) | 70% (1 FP) | 70% | 90% (2 FP) | 100% |
| white car (side) | 20% | 50% | 80% (1 FP) | 90% (1 FP) | 90% |
| grey car (top) | 30% | 50% | 95% | 100% | 100% |
| black car (top) | 100% | 100% | 100% | 100% | 100% |


# Speed comparison:

| Feature | Time (s)/vehicle |
| --- | --- |
| histogram | 0.002 |
| HOG | 0.026 |
| ResNet | 0.613 |
| OSNet | 0.240 |
