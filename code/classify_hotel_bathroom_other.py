import caffe
import os
from PIL import Image
import numpy as np
import csv

caffe.set_device(0)
caffe.set_mode_gpu()

net_model = '/project/focus/abby/hotelnet/models/hotelnet_google/deploy_hotel3.prototxt';
net_weights = '/project/focus/abby/hotelnet/models/hotelnet_google/hotelnet_iter_140000.caffemodel'
net = caffe.Net(net_model, net_weights, caffe.TEST);

im_path = '/project/focus/abby/multiclass/datasets/star_rating/train.txt'
with open(im_path,'rU') as f:
    reader = csv.reader(f,delimiter=' ')
    imList = list(reader)

origSize = 256
cropSize = 224
batchSize = 10

mean_im_path = '/project/focus/abby/hotelnet/models/places205CNN_mean.binaryproto'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_im_path, 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
mean_im = arr[0].mean(1).mean(1)

new_im_path = im_path.split('.')[0] + '_roomsonly.txt'
out_file = open(new_im_path,'a')

net.blobs['data'].reshape(batchSize,3,cropSize,cropSize)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_im)
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

pred = np.empty(len(imList))
for ix in range(0,len(imList)):
    try:
        orig_im = caffe.io.load_image(imList[ix][0])
        im_oversampled = caffe.io.oversample([caffe.io.resize_image(orig_im, (origSize,origSize))], (cropSize,cropSize))
        caffe_input = np.asarray([transformer.preprocess('data', crop) for crop in im_oversampled])
        net.blobs['data'].data[...] = caffe_input
        out = net.forward()
        prob = out['prob']
        pred[ix] = np.argmax(np.mean(prob,0))
        if pred[ix] == 1:
            out_file.write(imList[ix][0] + ' ' + imList[ix][1] + '\n')
            print 'Including ' + imList[ix][0]
        else:
            print 'Not including ' + imList[ix][0]
    except:
        continue

out_file.close()
