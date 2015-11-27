"""
Downloads the following:
- Stanford parser
- Stanford POS tagger
- Glove vectors
- SICK dataset (semantic relatedness task)
- Stanford Sentiment Treebank (sentiment classification task)

"""

from __future__ import print_function
import urllib2
import sys
import os
import shutil
import zipfile
import gzip

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib2.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.info().getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove dir path')
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz'
    filepath = download(url, dirpath)
    print('extracting ' + filepath)
    with gzip.open(filepath, 'rb') as gf:
        with open(filepath[:-3], 'w') as f:
            for line in gf:
                f.write(line)
    os.remove(filepath)



if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # data
    data_dir = os.path.join(base_dir, 'data')
    wordvec_dir = os.path.join(data_dir, 'glove')
   
    print("Downloading word vectors")
    download_wordvecs(wordvec_dir)
