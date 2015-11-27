# Download flickr8k data
wget -P data/flickr8k/ https://bitbucket.org/softmaxinc/image-captioner/downloads/dataset.json
wget -P data/flickr8k/ https://bitbucket.org/softmaxinc/image-captioner/downloads/googlenet_feats.json

# Download coco data train
wget -O data/coco/train/dataset.json https://softmaxstorage.blob.core.windows.net/coco-feats/coco_dataset_train.json
wget -O data/coco/train/googlenet_feats.txt https://softmaxstorage.blob.core.windows.net/coco-feats/googlenet_feats_train.txt

# Download coco data test
wget -O data/coco/test/dataset.json https://softmaxstorage.blob.core.windows.net/coco-feats/coco_dataset_val.json
wget -O data/coco/test/googlenet_feats.txt https://softmaxstorage.blob.core.windows.net/coco-feats/googlenet_feats_test.txt

# Convert coco dataset from cPickle to .txt
python scripts/imagefeats_to_torch.py

# Convert coco train dataset from .txt to .th
image_dir="data/coco/train"
image_pre="googlenet_feats"
dataset_pre = "dataset"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats_txt.lua $image_dir/$image_pre.txt \
        $image_dir/$image_pre.th
fi

if [ ! -f $image_dir/$dataset_pre.th ]; then
    th scripts/convert_dataset.lua $image_dir/$dataset_pre.txt \
        $image_dir/$dataset_pre.th
fi

# Convert coco test dataset from .txt to .th
image_dir="data/coco/test"
image_pre="googlenet_feats"
dataset_pre = "dataset"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats_txt.lua $image_dir/$image_pre.txt \
        $image_dir/$image_pre.th
fi

if [ ! -f $image_dir/$dataset_pre.th ]; then
    th scripts/convert_dataset.lua $image_dir/$dataset_pre.txt \
        $image_dir/$dataset_pre.th
fi

# Preprocess captioning data
python scripts/download.py
python scripts/create_vocab.py

# Preprocess flickr vocab vectors
glove_dir="data/flickr8k"
glove_pre="vocab_feats"
glove_dim="600d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/convert_wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi

# Preprocess coco vocab vectors
glove_dir="data/coco"
glove_pre="vocab_feats"
glove_dim="600d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/convert_wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi


# Convert features to torch format
image_dir="data/flickr8k"
image_pre="googlenet_feats"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats.lua $image_dir/$image_pre.json \
        $image_dir/$image_pre.th
fi

# Preprocess glove vectors
glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/convert_wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi

# Convert features to torch format
image_dir="data/coco"
image_pre="train.txt"

if [ ! -f $image_dir/$image_pre.th ]; then
    th scripts/convert_imagefeats.lua $image_dir/$image_pre.txt \
        $image_dir/$image_pre.th
fi

# Preprocess items for calculating bleu score
th main.lua

# Install dependencies
luarocks install torch
luarocks install torch7
luarocks install nn
luarocks install nngraph
luarocks install optim
luarocks install cutorch
luarocks install cunn
luarocks install json
luarocks install dkjson

# Server info
luarocks install async
luarocks install buffer
luarocks install http
luarocks install luasocket

# For loading vision modules
luarocks install loadcaffe