# simpleNALU-tf

This is a simple implementation of DeepMinds Neural Arithmetic Logic Units (NALU) paper in TensorFlow.
https://arxiv.org/abs/1808.00508


## Usage 

#### 1. Install the dependencies
```bash
$ pip install -r requirements.txt
```

#### 2. Train the model
```bash
$ python NALU.py
```

#### 3. Open the TensorBoard 
To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ on your web browser.
```bash
$ tensorboard --logdir='tmp' --port=6006
```


## Contribution

1.  Clone/fork
2.  Push up to a new branch
3.  Submit a PR to the upstream `master` branch.
