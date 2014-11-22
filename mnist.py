import numpy as np
import struct, gzip, os, sys
from collections import namedtuple

_data = namedtuple('data',['X','Y'])
_datadir = os.path.join(os.path.dirname(__file__),'data')

def _read_header(filename, fmt, numBytes):
	f = gzip.GzipFile(filename,'rb')
	header = struct.unpack(fmt,f.read(numBytes))
	buff = f.read()
	f.close()
	return header,buff

def _labels_datatype():
	return np.dtype('>b')

def _images_datatype():
	return np.dtype('>B')

def _load_mnist(filenames):
	images_header,images_buffer = _read_header(filenames[0],'>IIII',16)
	labels_header,labels_buffer = _read_header(filenames[1],'>II',8)
	labels = np.fromstring(labels_buffer,dtype=_labels_datatype(),count=labels_header[1])
	images = np.fromstring(images_buffer,dtype=_images_datatype(),count=images_header[1]*images_header[2]*images_header[3]).reshape(images_header[1],images_header[2],images_header[3])
	return _data(X=images,Y=labels)

def traindataset():
	return _load_mnist((os.path.abspath(os.path.join(_datadir,'train-images-idx3-ubyte.gz')),
		os.path.abspath(os.path.join(_datadir,'train-labels-idx1-ubyte.gz'))))


def testdataset():
	return _load_mnist((os.path.abspath(os.path.join(_datadir,'t10k-images-idx3-ubyte.gz')),
		os.path.abspath(os.path.join(_datadir,'t10k-labels-idx1-ubyte.gz'))))