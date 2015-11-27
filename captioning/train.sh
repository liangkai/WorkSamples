export PATH=/Users/david/torch/install/bin:$PATH

num_layers=${1-1}
mem_dim=${2-2}
emb_dim=${3-2}
regularization=${4-1e-5}
learning_rate=${5-0.1}
optim_method=${6-rmsprop}
in_dropout_prob=${7-0.2}
hidden_dropout_prob=${8-0.5}
beam_size=${9-6}
combine_module=${10-embedlayer}
hidden_module=${11-projlayer}
dataset=${12-coco} #${12-flickr8k}
dropout=${13-}
gpu_mode=${14-}


th image_captioning/main.lua \
-batch_size 100 \
-dataset $dataset \
-mem_dim $mem_dim \
-emb_dim $emb_dim \
-epochs 100 \
-num_layers $num_layers \
-combine_module $combine_module \
-hidden_module $hidden_module \
-in_dropout_prob $in_dropout_prob \
-hidden_dropout_prob $hidden_dropout_prob \
-learning_rate $learning_rate \
-beam_size $beam_size \
$dropout $gpu_mode \
-optim $optim_method

# First argument is memory dimensions
# Second argument is embed dimensions
# Third argument is number of layers
# Fourth argument is learning rate
# Fifth argument is dropout
# Sixth argument is gpu_mode
# -dropout \
#-load_model \
#-model_epoch 98 \
# -gpu_mode \
