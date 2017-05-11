# Person Search with Natural Language Description

This project aims at searching person using natural language description. Mainly based on our CVPR 2017 paper [Person Search with Natural Language Description](https://arxiv.org/pdf/1702.05729.pdf). The code is modified from the [Neuraltalk2](https://github.com/karpathy/neuraltalk2) written by Andrej Karpathy.


## Installation

This code is written in Lua and requires Torch. See the [Torch](http://torch.ch/) installation documentation for more details. 
To run this code, the following packages must be installed:

- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- [cjson](https://www.kyne.com.au/~mark/software/lua-cjson-manual.html)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)


## Data Preparation

1. Request the dataset from sli [at] ee.cuhk.edu.hk or xiaotong [at] ee.cuhk.edu.hk (academic only).

2. Save data into a json file that contains a list of image paths, person ID, and raw captions for each image, of the form:

  ```
  [{"split": "train", "captions": ["A woman is wearing a gray shirt, a pair of brown pants and a pair of shoes.", "She is wearing a dark grey top and light colored pants."], "file_path": "train_query/p11600_s14885.jpg", "id": 11003}]
  ```

  You can also directly download the [json file](https://drive.google.com/open?id=0B-GOvBat1maOUVZIRnhHN2oyQUE) and save it to "data/reid_raw.json"
  
3. Run prepro_data.py as follows:
   
   ```
   python prepro_data.py --input_json data/reid_raw.json --images_root data/imgs --max_length 50 --word_count_threshold 2 --output_json data/reidtalk.json --output_h5 data/reidtalk.h5
   ```

## Testing

1. Download the pretrained model from [snapshot](https://drive.google.com/open?id=0B-GOvBat1maOSUllWW9OSDkwcEE).

2. Run Retrieval.lua. The result should be around:
   
   ```Shell
   top- 1 = 19.71%
   top- 5 = 43.20%
   top-10 = 55.24%
   ```

## Training

1. Download the VGG-16 network pre-trained model from [model](https://drive.google.com/open?id=0B-GOvBat1maOLWh1Y09Tamw4M2M).

2. Run train.lua.

## Citation
    @article{li2017person,
      title={Person search with natural language description},
      author={Li, Shuang and Xiao, Tong and Li, Hongsheng and Zhou, Bolei and Yue, Dayu and Wang, Xiaogang},
      journal={arXiv preprint arXiv:1702.05729},
      year={2017}
    }
