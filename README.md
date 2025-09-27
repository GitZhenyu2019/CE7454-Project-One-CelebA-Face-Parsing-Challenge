# CelebA Face Parsing Challenge Solution

## Description of the files

- `train.py`  
  Entry point for training. Loads data, builds the model, and trains it.

- `trainer.py`  
  Defines the `Trainer` class which handles model initialization, training loop, optimizer, scheduler, and checkpoint saving.

- `tester.py`  
  Defines the `Tester` class for running inference on validation/test images and saving predicted masks.

- `test.py`  
  Entry point for evaluation. Uses `Tester` to predict masks for the validation/test set.

- `model.py`  
  Contains the implementation of the UNet model.

- `data_loader.py`  
  Dataset and dataloader utilities for loading images and corresponding masks.

- `utils.py`  
  Helper functions, including loss functions (cross entropy + dice), scheduler, saving masks, and counting parameters.

- `parameter.py`  
  Contains the configuration function `get_parameters()` which stores hyperparameters and paths.

- `solution/ckpt.pth`  
  Trained model checkpoint file.

- `Readme.txt`  
  This file.

---

## References to third-party libraries

- [PyTorch](https://pytorch.org/)  
- [torchvision](https://pytorch.org/vision/stable/index.html)  
- [Pillow (PIL)](https://python-pillow.org/)  
- [numpy](https://numpy.org/)  

---

## Instructions for Testing

1. **Training (optional)**  
   Run:
   ```bash
   python train.py



