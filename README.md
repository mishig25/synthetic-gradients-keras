# Decoupled Neural Interfaces using Synthetic Gradients

Keras as an interface to Tensorflow implementation of Decoupled Neural Interfaces using Synthetic Gradients.

Link to the paper:  [https://arxiv.org/abs/1608.05343](https://arxiv.org/abs/1608.05343)

<img src="https://storage.googleapis.com/deepmind-live-cms/documents/3-6.gif" width="200">

GIF demonstrating decoupled learning through synthetic gradients. Source: [DeepMind blog post](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/) by Max Jaderberg.

### Contents:
- `main.py` - main function
- `model.py` - synthetic grads implementation
- `demo_nb.ipynb` - jupyter notebook for demonstrating contents and usage of `model.py`

### Prerequisites:
- Python 3.6
- Keras 2.2.0
- Tensorflow 1.8.0

### Usage:
First option:
```
main.py [-h] [-I ITERATIONS] [-B BATCH] [-P UPDATE_PROB] [-L L_RATE]

optional arguments:
  -h, --help            show this help message and exit
  -I ITERATIONS, --iterations ITERATIONS
                        Number of Iterations: int
  -B BATCH, --batch BATCH
                        Batch Size: int
  -P UPDATE_PROB, --update_prob UPDATE_PROB
                        Synthetic Grad Update Probability: float [0,1]
  -L L_RATE, --l_rate L_RATE
                        Learning Rate: float
```
Second option:
```
Use Jupyter Lab or Notebooks to open `demo_nb.ipynb`
```

### Tested on:
- OS: ubuntu 16.04 LTS
- GPU: single GeForce GTX 1070 

### Results

|  | Accuracy | Loss |
|------|-------|-------|
|MNIST| 0.917 | 0.288 |

### References
- [Guide by Andrew Trask](https://iamtrask.github.io/2017/03/21/synthetic-gradients/)
- [Tensorflow Implementation](https://github.com/nitarshan/decoupled-neural-interfaces)
- [PyTorch Implementation](https://github.com/andrewliao11/dni.pytorch)