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
usage: main.py [-h] -i INPUT -r REFERENCE_IMG [-f {hog,histogram,dnn}]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input video
  -r REFERENCE_IMG, --reference_img REFERENCE_IMG
                        path to reference image
  -f {hog,histogram,dnn}, --feature {hog,histogram,dnn}
                        Feature type to use for comparison
```
