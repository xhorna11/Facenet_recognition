"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import os
import struct
import argparse
import facenet
import align.detect_face
import time
import cv2

def evaluate(args,distance):

    subdirs=listdir(args) 
    names=[];
    features=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            f=open(join(subdir,file),"rb")
            data=[];
            for i in range(0,128):
                a=f.read(4)
                b=struct.unpack("f",a)
                data.append(b[0])
            f.close()
            features.append(data)
            names.append(name)
    TP=0
    FP=0
    TN=0
    FN=0
    total=0;
    for n, tst in enumerate(features):
        for i, ref in enumerate(features):
            if (i<=n):
                continue
            face_distance = np.sqrt(np.sum(np.square(np.subtract(features[n], features[i]))))
            if (face_distance<distance):
                if (names[n]==names[i]):
                    TP+=1
                else:
                    FP+=1
                    print("FP names ",names[n]," ",names[i]) 
            else:
                if (names[n]==names[i]):
                    FN+=1
                    print("FN names ",names[n]," ",names[i]) 
                else:
                    TN+=1
            total=total+1;
    print("TP=",TP," FP=",FP," FN=",FN," TN=",TN," Total=",total)
    return 0

def getReferences(dir):
    subdirs=listdir(dir) 
    names=[];
    image_filenames=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(dir,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=join(subdir,file);
            image_filenames.append(file)
            names.append(name)
    images = load_and_align_data(image_filenames,160,44,1.0)
    models_path="models/facenet/20170512-110547"
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(models_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb,names

def createDetector(gpu_memory_fraction):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet,rnet,onet

def alignImages(images, image_size, margin,pnet,rnet,onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_samples = len(images)
    img_list =[] # [None] * nrof_samples
    for i in range(nrof_samples):
        img = images[i]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if (len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            #img_list[i] = prewhitened
            img_list.append(prewhitened)
    if (len(img_list)):
        images = np.stack(img_list)
    else:
        images=[]
    return images

def proccessVideo(dir,dist):
    models_path="models/facenet/20170512-110547"
    pnet,rnet,onet=createDetector(1.0)
            
    [references,names]=getReferences(dir+"reference")
    dir=dir+"test"
    subdirs=listdir(dir) 
    TP=0
    FP=0
    FN=0
    TN=0
    total=0
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(models_path)
            for subdir in (subdirs):
                name=subdir
                subdir=join(dir,subdir)
                files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
                for file in (files):
                    file=join(subdir,file);
                    videoCapture=cv2.VideoCapture(file)
                    if (not videoCapture.isOpened()):
                        continue
                    images=[]
                    ret,frame=videoCapture.read()
                    while(ret):
                        images.append(frame)
                        ret,frame=videoCapture.read()
                    images=alignImages(images,160,44,pnet,rnet,onet)
                    if (not len(images)):
                        continue
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    # Run forward pass to calculate embeddings
                    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    
                    for n, tst in enumerate(emb):
                        for i, ref in enumerate(references):
                            face_distance = np.sqrt(np.sum(np.square(np.subtract(tst, ref))))
                            if (face_distance<dist):
                                if (name==names[i]):
                                    TP+=1
                                else:
                                    FP+=1
                            else:
                                if (name==names[i]):
                                    FN+=1
                                else:
                                    TN+=1
                            total=total+1
                    print("TP= ",TP ," FP= ",FP," FN= ",FN," TN= ",TN," total= ", total)

def main(args,distance,eval):
   
    proccessVideo("C:/Users/Veronika/Desktop/video/",distance)
    return 0
    if (eval):
        evaluate("C:/Users/Veronika/Desktop/facenet-master/src/align/datasets/features",distance)
        return 0
    models_path="models/facenet/20170512-110547"
    subdirs=listdir(args) 
    names=[];
    image_filenames=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=join(subdir,file);
            image_filenames.append(file)
            names.append(name)

    #images = load_and_align_data(image_filenames,160,44,1.0)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            filenames=[]
            offset=0
            # Load the model
            facenet.load_model(models_path)
            for n in range(len(image_filenames)):
                filenames.append(image_filenames[n])
                if (len(filenames)==200 or n==(len(image_filenames)-1)):
                    
                    # Load data
                    images =facenet.load_data(filenames,False,False,160)
                    start=time.time()
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    # Run forward pass to calculate embeddings
                    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    stop=time.time()
                    print("Time = ",(stop-start)/200)
                    nrof_images = len(filenames)
                    data_path="C:/Users/Veronika/Desktop/facenet-master/src/align/datasets/features"
                    for i in range(nrof_images):
                        subdir=join(data_path,names[i+offset])
                        if not os.path.exists(subdir):
                            os.makedirs(subdir)
                        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
                        idx=len(files)
                        filename=str(idx+1) + ".dat"
                        file=open(join(subdir,filename),"wb")
                        file.write(emb[i,:])
                        file.close()
                    print("Proccessed from ",offset,"to ",n,"\n")
                    offset=n
                    filenames.clear()
      
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


if __name__ == '__main__':
    main("C:/Users/Veronika/Desktop/facenet-master/src/align/datasets/lfw_mtcnnpy_160",1,True)
