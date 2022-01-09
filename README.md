# FlatGCN
This is the official Pytorch-version code of FlatGCN (Flattened Graph Convolutional Networks for Recommendation, submitted to ICASSP2022).

## Requirements
python >= 3.7

pytorch == 1.9.1

pickle == 0.7.5

scikit-learn == 0.24.2

pandas == 1.3.3

numpy == 1.21.2

scipy == 1.7.1

## Usage
We provide two preprocessed experimental datasets (LastFM, Yelp2018) in **data** folder. For the Yelp2018 dataset, because the data-mapping file (yelp2018_map.pkl) is too large to directly uploaded (exceeds git's 100M file upload limitation), we store it in Google Cloud Disk, the access link is as follows:
<https://drive.google.com/drive/folders/1-5pKyLY1XOLwswsppcIjQnWu0MigiTjj?usp=sharing>

For **LastFM** dataset, you can use the following run commands (optional *Meta2Vec* embedding or *LightGCN* embedding):
```
python main.py --dataset lastfmUA --emb lgn --model FlatGCN
python main.py --dataset lastfmUA --emb n2v --model FlatGCN
```

For **Yelp2018** dataset, you need to first download the data-mapping file from the above link and place it in the **data** folder, then you can use the following run commands (optional *Meta2Vec* embedding or *LightGCN* embedding):
```
python main.py --dataset yelp2018 --emb lgn --model FlatGCN
python main.py --dataset yelp2018 --emb n2v --model FlatGCN
```
