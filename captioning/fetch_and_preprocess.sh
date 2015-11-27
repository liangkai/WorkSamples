export PATH=/Users/david/torch/install/bin:$PATH

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

