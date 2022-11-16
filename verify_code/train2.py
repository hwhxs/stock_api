from PIL import Image
import os
import sys
from numpy import * 
import operator
from sklearn import datasets,neighbors


# convert image to vector  
def  img2vector(filename):  
    rows = 36  
    cols = 16 
    imgVector = [] 
    fileIn = open(filename)  
    for row in range(rows):  
        lineStr = fileIn.readline()  
        for col in range(cols):  
            imgVector.append(int(lineStr[col])) 
    return imgVector 

def img_split(img_name, img_path, code, dir):
  img = Image.open(img_path)
  img = img.convert('L')
  WHITE, BLACK = 1, 0
  img = img.point(lambda x: WHITE if x > 128 else BLACK)
  i = 0
  for pos in [[13, 0, 30, 36], [31, 0, 47, 36], [48, 0, 65, 36], [66, 0, 83, 36]]:
    split_img = img.crop((pos[0], pos[1], pos[2], pos[3]))
    data = asarray(split_img)
    savetxt(dir+'/'+code[i:i+1]+"_"+str(i)+'_'+img_name+'.txt', data, fmt="%d", delimiter='')
    i = i+1
  pass


def classify(sample,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]     #数据集行数即数据集记录数
    '''距离计算'''
    diffMat=tile(sample,(dataSetSize,1))-dataSet         #样本与原先所有样本的差值矩阵
    sqDiffMat=diffMat**2      #差值矩阵平方
    sqDistances=sqDiffMat.sum(axis=1)       #计算每一行上元素的和
    distances=sqDistances**0.5   #开方
    sortedDistIndicies=distances.argsort()      #按distances中元素进行升序排序后得到的对应下标的列表
    '''选择距离最小的k个点'''
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    '''从大到小排序'''
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__=="__main__":
  os.chdir('/root/verify_code/')
  work_dir = './train_imgs/'
  for parent, dirnames, filenames in os.walk(work_dir):
    for filename in filenames:
      file_path = os.path.join(parent, filename)
      img_split(filename, file_path, filename.split('_')[0],'split_imgs')

  X = []
  y = []
  kn_clf = neighbors.KNeighborsClassifier()
  spilt_imgs_dir = './split_imgs/'
  for parent, dirnames, filenames in os.walk(spilt_imgs_dir):
    for filename in filenames:
      file_path = os.path.join(parent, filename)
      X.append(img2vector(file_path))
      y.append(filename.split('_')[0])
  kn_clf.fit(array(X),y)
  predict_img = sys.argv[1]
  img_split('code.jpg', predict_img, '9999', 'tmp')
  spilt_imgs_dir = './tmp/'
  prdict_code = ''
  for i in range(4):
    #prdict_code = prdict_code+classify(img2vector("tmp/9_"+str(i)+'_code.jpg.txt'), array(X), y, 1)
    # prdict_code = prdict_code+kn_clf.predict(array([img2vector("tmp/9_"+str(i)+'_code.jpg.txt')]))
    print(kn_clf.predict(array([img2vector("tmp/9_"+str(i)+'_code.jpg.txt')])))
  print(prdict_code)

