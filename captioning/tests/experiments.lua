require 'loadcaffe'

model = loadcaffe.load('googlenet_deploy.prototxt', 'googlenet.caffemodel', 'ccn2')