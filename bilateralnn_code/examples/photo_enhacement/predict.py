import numpy as np
from config import *
import cv2
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100

    print "np.max(img1): ", np.max(img1), "np.max(img2): ",np.max(img2), "np.max(img1 - img2): ", np.max(img1 - img2)
    print "math.sqrt(mse):", math.sqrt(mse)
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def predict(prototxt, caffe_model):

    net = caffe.Net(prototxt, caffe_model, caffe.TEST)
    net.forward()
    result = net.blobs['image_upsampled'].data
    print result[0].shape

    for i in range(result.shape[0]):
        cv2.imwrite('outputs/output_image_' + str(i) + '.jpg', np.rollaxis(net.blobs['image_output'].data[i], 0, 3))
        cv2.imwrite('outputs/output_image_' + str(i) + '.jpg', np.rollaxis(net.blobs['image_upsampled'].data[i], 0, 3))

        print "PSNR for image:", i, psnr(net.blobs['image_output'].data[i], net.blobs['image_upsampled'].data[i])

    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' <prototxt> <caffe_model>')
    else:
        predict(sys.argv[1], sys.argv[2])
