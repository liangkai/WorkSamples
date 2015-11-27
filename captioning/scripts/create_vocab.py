"""
Preprocessing script for SICK data.

"""

import json
import os
import glob
import time
import numpy as np 

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def build_vocab(dataset_path, dst_path, feat_path, word_count_threshold = 5):
  dataset = json.load(open(dataset_path, 'r'))
  sentence_iterator = dataset['images']

  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation
  print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
  t0 = time.time()
  word_counts = {}
  nsents = 0
  for sentences in sentence_iterator:
    for sent in sentences['sentences']:
      nsents += 1
      for w in sent['tokens']:
        word_counts[w] = word_counts.get(w, 0) + 1

  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print ('filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0))

  # with K distinct words:
  # - there are K+1 possible inputs (START token and all the words)
  # - there are K+1 possible outputs (END token and all the words)
  # we use ixtoword to take predicted indeces and map them to words for output visualization
  # we use wordtoix to take raw words and get their index in word vector matrix
  ixtoword = {}
  ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
  wordtoix = {}
  wordtoix['#START#'] = 0 # make first vector be the start token
  ix = 1
  for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

  # compute bias vector, which is related to the log probability of the distribution
  # of the labels (words) and how often they occur. We will use this vector to initialize
  # the decoder weights, so that the loss function doesnt show a huge increase in performance
  # very quickly (which is just the network learning this anyway, for the most part). This makes
  # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
  # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
  word_counts['.'] = nsents
  bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
  bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
  bias_init_vector = np.log(bias_init_vector)
  bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

  with open(feat_path, "w") as f:
    for i in range(0, len(bias_init_vector)):
      curr_vals = [bias_init_vector[i]] * 600
      f.write(ixtoword[i] + " ")
      f.write(" ".join(map(str, curr_vals)))
      if i != len(bias_init_vector) - 1:
        f.write("\n")

  with open(dst_path, 'w') as f:
    for i in ixtoword:
      w = ixtoword[i]
      f.write(w + '\n')
      
  print('saved vocabulary to %s' % dst_path)
  print('saved features to %s' % feat_path)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing Flickr8k Caption dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    flickr8k_dir = os.path.join(data_dir, 'flickr8k')

    # get vocabulary
    build_vocab(
        os.path.join(flickr8k_dir, 'dataset.json'),
        os.path.join(flickr8k_dir, 'vocab.txt'),
        os.path.join(flickr8k_dir, 'vocab_feats.600d.txt'))

    print('=' * 80)
    print('Preprocessing Coco Caption dataset')
    print('=' * 80)
    coco_dir = os.path.join(data_dir, 'coco') 

    # get vocabulary
    build_vocab(
        os.path.join(coco_dir, 'train/dataset.json'),
        os.path.join(coco_dir, 'vocab.txt'),
        os.path.join(coco_dir, 'vocab_feats.600d.txt'))

