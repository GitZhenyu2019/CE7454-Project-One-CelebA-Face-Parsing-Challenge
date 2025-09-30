# CelebA Face Parsing Challenge Solution

## Description of the files

- `solution/train.py`  
  Entry point for training. Loads data, builds the model, and trains it.

- `solution/trainer.py`  
  Defines the `Trainer` class which handles model initialization, training loop, optimizer, scheduler, and checkpoint saving.

- `solution/tester.py`  
  Defines the `Tester` class for running inference on validation/test images and saving predicted masks.

- `solution/test.py`  
  Entry point for evaluation. Uses `Tester` to predict masks for the validation/test set.

- `solution/model_utils.py`  
  Defines components of UNet used by `model.py`.

- `solution/run.py`  
  Main entry point for inference as required by the competition. Loads a trained checkpoint and performs segmentation on a single input image.

- `solution/model.py`  
  Contains the implementation of the UNet model.

- `solution/data_loader.py`  
  Dataset and dataloader utilities for loading images and corresponding masks.

- `solution/utils.py`  
  Helper functions, including loss functions (cross entropy + dice), scheduler, saving masks, and counting parameters.

- `solution/parameter.py`  
  Contains the configuration function `get_parameters()` which stores hyperparameters and paths.

- `solution/color.py`
  Colorize single channel gray scale masks to RGB masks through a color palette.

- `solution/ckpt.pth`  
  Trained model checkpoint file.

- `Readme.txt`  
  This file.

---

## References to third-party libraries

- [PyTorch](https://pytorch.org/)  
- [torchvision](https://pytorch.org/vision/stable/index.html)  
- [Pillow (PIL)](https://pypi.org/project/pillow/)  
- [numpy](https://numpy.org/)  

---

## Instructions for Testing

```bash
- single image prediction

pip install -r solution/requirements.txt
python solution/run.py --input /path/to/input.jpg --output /path/to/output.png --weights solution/ckpt.pth

- multiple images prediction

python solution/test.py --val_images /path/to/input --masks_out_dir /path/to/output






