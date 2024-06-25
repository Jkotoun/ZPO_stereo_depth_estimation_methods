# Depth estimation
This project compares performance of several depth estimation methods:
- Traditional stereo depth estimation methods - Block Matching (implemented from scratch) and Semi-Global Block Matching (opencv implementation)
- HitNet Neural Network for stereo depth estimation
- Depth Anything Neural Network for monocular depth estimation
  
The results can be found in ZPO.pdf file.

# Team members
- Josef Kotoun
- Jiří Vlasák
- Vít Tlustoš

# Installation
Install requirements with `pip install -r requirements.txt`

# Usage
To see predictions on example image, run `python demo.py`. The script will display the disparity map for the example image for all of the models.

To run the DepthAnything model on input image, run `python run.py --img img.png`. The script will display the predicted depth map for the input image.


To evaluate the models run the `predict.py` script. It expects path to the dataset, patterns for left and right images and ground truth disparity maps. Patterns for ground truth are not needed if the `--evaluate` flag is not set. To evaluate the models on the Middlebury dataset, download the dataset and ground truths from http://vision.middlebury.edu/stereo/submit3/ and set the `--data_pattern` to the path of the dataset.
 The script can be used to evaluate the following models:

EXAMPLE USAGE:
```bash
SMGB 
python predict.py --data_pattern="data/ --iml_pattern="**\im0.png" --imr_pattern="**\im1.png" --gtl_pattern="**/disp0GT.pfm" --evaluate --predictor="sgbm" 

BM
python predict.py --data_pattern="data/" --iml_pattern="**\im0.png" --imr_pattern="**\im1.png" --gtl_pattern="**/disp0GT.pfm" --evaluate --predictor="bm"

HITNET
python predict.py --data_pattern="data/" --iml_pattern="**\im0.png" --imr_pattern="**\im1.png" --gtl_pattern="**/disp0GT.pfm" --evaluate --predictor="hitnet" --model_path="models/flyingthings_finalpass_xl.pb" 

```
