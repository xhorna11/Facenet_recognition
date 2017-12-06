import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import os

def evaluate(args,distance):

    subdirs=listdir(args) 
    names=[];
    features=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=open(join(subdir,file),"rb")
            data=file.read()
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
            face_distances = np.sqrt(np.sum(np.square(np.subtract(features[i,:], features[j,:]))))
            if (face_distance<distance):
                if (names[n]==names[i]):
                    TP+=1
                else:
                    FP+=1
            else:
                if (names[n]==names[i]):
                    FN+=1
                else:
                    TN+=1
            total=total+1;
    print("TP=",TP," FP=",FP," FN=",FN," TN=",TN," Total=",total)
    input("Press Enter to continue ...")
    return 0

#if __name__ == '__main__':
  #main("C:/Users/Veronika/Desktop/facenet-master/src/align/datasets/features",0.5)