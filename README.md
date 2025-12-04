# Adaptive Memory-Augmented Transformer for Unified Facial Landmark Detection


## 1. Overview

<p align="center">
    <img src="./Images/AM-Former.jpg"/> <br />
    <em> 
    Figure 2: The architecture of the proposed AM-Former for UFLD. A cyclic data iterator enables unified training across 
    heterogeneous datasets (300W, AFLW, COFW, WFLW). A ResNet-50 backbone with multi-scale fusion generates input tokens, which 
    are fed into the Memory-Augmented Encoder (MAE) to dynamically refine common memory slots under the Memory Regularization Loss (MRL). 
The decoded features are finally mapped by dataset-specific Label (FC) and Coord (MLP) heads to landmark labels and coordinates.
    </em>
</p>


## 2. File Structure


    AM-Former
    ├── Images
    │   └── AM-Former.jpg
    ├── README.md
    ├── lib
    │   ├── config
    │   │   ├── __init__.py
    │   │   ├── default.py
    │   │   └── models.py
    │   ├── core
    │   │   ├── evaluate.py
    │   │   ├── function.py
    │   │   ├── inference.py
    │   │   └── loss.py
    │   ├── dataset
    │   │   ├── __init__.py
    │   │   ├── aflw.py
    │   │   ├── cofw.py
    │   │   ├── face300w.py
    │   │   └── wflw.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── attention.py
    │   │   ├── backbone.py
    │   │   ├── hrnet.py
    │   │   ├── matcher.py
    │   │   ├── new_containers.py
    │   │   ├── pem.py
    │   │   ├── pose_transformer.py
    │   │   ├── positional_encoding.py
    │   │   └── transformer.py
    │   ├── utils
    │   │   ├── __init__.py
    │   │   ├── metrics.py
    │   │   ├── transforms.py
    │   │   ├── utils.py
    │   │   └── vis.py
    ├── requirements.txt
    ├── tools
    │   ├── Test.py
    │   ├── Test_wflw.py
    │   ├── _init_paths.py
    │   ├── trace.py
    │   └── train.py
    └── wflw.yaml


## 3. Usage

### 1. Configuring your environment (Prerequisites):
    
Installing necessary packages: `pip install -r requirements.txt`.
    
### 2. Preparing the Datasets:

We propose a multi-dataset collaborative training approach for facial landmark detection, which constructs parallel iterators for four mainstream datasets (AFLW, WFLW, 300W, and COFW), achieves alignment of heterogeneous annotation protocols across datasets under a unified adaptive memory-augmented Transformer framework, enabling joint training of multiple datasets while optimizing cross-dataset generalization performance. The structure of our dataset is as follows:


    dataset
    ├── 300W
    │   ├── annotations
    │   │   ├── face_landmarks_300w_test.json
    │   │   ├── face_landmarks_300w_train.json
    │   │   ├── face_landmarks_300w_valid.json
    │   │   ├── face_landmarks_300w_valid_challenge.json
    │   │   └── face_landmarks_300w_valid_common.json
    │   ├── ibug
    │   ├── test.tsv
    │   ├── test_common.tsv
    │   ├── test_ibug.tsv
    │   ├── testset
    │   ├── train.tsv
    │   └── trainset
    ├── AFLW
    │   ├── face_landmarks_aflw_test.csv
    │   ├── face_landmarks_aflw_train.csv
    │   ├── flickr
    │   │   ├── 0
    │   │   ├── 2
    │   │   └── 3
    ├── COFW
    │   ├── COFW_test_color.mat
    │   ├── COFW_train_color.mat
    ├── WFLW
    │   ├── WFLW_annotations
    │   │   ├── list_98pt_rect_attr_train_test
    │   │   │   ├── README
    │   │   │   ├── list_98pt_rect_attr_test.txt
    │   │   │   ├── list_98pt_rect_attr_train.txt
    │   │   ├── list_98pt_test
    │   │   │   ├── README
    │   │   │   ├── list_98pt_test.txt
    │   │   │   ├── list_98pt_test_blur.txt
    │   │   │   ├── list_98pt_test_expression.txt
    │   │   │   ├── list_98pt_test_illumination.txt
    │   │   │   ├── list_98pt_test_largepose.txt
    │   │   │   ├── list_98pt_test_makeup.txt
    │   │   │   └── list_98pt_test_occlusion.txt
    │   ├── WFLW_images
    │   │   ├── 0--Parade
    │   │   ├── 1--Handshaking
    │   │   ├── 10--People_Marching
    │   │   ├── 11--Meeting
    │   │   ├── 12--Group
    │   │   ├── 13--Interview
    │   │   ├── 14--Traffic
    │   │   ├── 15--Stock_Market
    │   │   ├── 16--Award_Ceremony
    │   │   ├── 17--Ceremony
    │   │   ├── 18--Concerts
    │   │   ├── 19--Couple
    │   │   ├── 2--Demonstration
    │   │   ├── 20--Family_Group
    │   │   ├── 21--Festival
    │   │   ├── 22--Picnic
    │   │   ├── 23--Shoppers
    │   │   ├── 24--Soldier_Firing
    │   │   ├── 25--Soldier_Patrol
    │   │   ├── 26--Soldier_Drilling
    │   │   ├── 27--Spa
    │   │   ├── 28--Sports_Fan
    │   │   ├── 29--Students_Schoolkids
    │   │   ├── 3--Riot
    │   │   ├── 30--Surgeons
    │   │   ├── 31--Waiter_Waitress
    │   │   ├── 32--Worker_Laborer
    │   │   ├── 33--Running
    │   │   ├── 34--Baseball
    │   │   ├── 35--Basketball
    │   │   ├── 36--Football
    │   │   ├── 37--Soccer
    │   │   ├── 38--Tennis
    │   │   ├── 39--Ice_Skating
    │   │   ├── 4--Dancing
    │   │   ├── 40--Gymnastics
    │   │   ├── 41--Swimming
    │   │   ├── 42--Car_Racing
    │   │   ├── 43--Row_Boat
    │   │   ├── 44--Aerobics
    │   │   ├── 45--Balloonist
    │   │   ├── 46--Jockey
    │   │   ├── 47--Matador_Bullfighter
    │   │   ├── 48--Parachutist_Paratrooper
    │   │   ├── 49--Greeting
    │   │   ├── 5--Car_Accident
    │   │   ├── 50--Celebration_Or_Party
    │   │   ├── 51--Dresses
    │   │   ├── 52--Photographers
    │   │   ├── 53--Raid
    │   │   ├── 54--Rescue
    │   │   ├── 55--Sports_Coach_Trainer
    │   │   ├── 56--Voter
    │   │   ├── 57--Angler
    │   │   ├── 58--Hockey
    │   │   ├── 59--people--driving--car
    │   │   ├── 61--Street_Battle
    │   │   ├── 7--Cheering
    │   │   ├── 8--Election_Campain
    │   │   └── 9--Press_Conference
    │   ├── face_landmarks_wflw_test.csv
    │   └── face_landmarks_wflw_train.csv

You can download the datasets from the following links:
   + [AFLW](https://drive.google.com/drive/folders/1CG-6OxaJpkwmzHNnPHEAU7y_fVjMjIAl?usp=sharing),
   + [300W](https://drive.google.com/drive/folders/1uNnaVCoB5cUaT6JZVQtSU5FcYA967GWk?usp=sharing)
   + [COFW](https://drive.google.com/drive/folders/1lHvLzYnsqziZw7FI8AbmCOfmZSYvIZ0b?usp=sharing)
   + [WFLW](https://drive.google.com/drive/folders/1shd-yo8OkNzi9HkGulUI4v6lG38qUOWs?usp=sharing)

### 3. Training the Model:

Our training pipeline implements multi-dataset collaborative training. To start training:

`python tools/train.py --cfg wflw.yaml`


### 4. Testing the Model

To evaluate the trained model on specific datasets:

``` 
# Test on WFLW dataset
python tools/Test_wflw.py --cfg wflw.yaml --model_path path/to/your/best_model.pth
    
# Test on 300W, COFW, AFLW datasets
python tools/Test.py --cfg wflw.yaml --model_path path/to/your/best_model.pth
```