# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F

import torch.optim as optim  #优化器
import numpy as np
from matplotlib import pyplot as plt

import cv2
from PIL import Image ,ImageDraw ,ImageFont

import os,random,math,shutil


epochs = 20
learning_rate = 0.002005
'''转换图片格式'''
transform = transforms.Compose([#transforms.Resize(32,32), #图片大小调整
                               transforms.ToTensor(),     #数据类型调整
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]#归一化处理
                               )
classes = ('0','1','2','3','4','5','6','7','8','9','A','B',
           'C','D','E','F','G','H','J','K','L','M','N','P',
           'Q','R','S','T','U','V','W','X','Y','Z',
           '川','鄂', '赣', '甘', '贵' ,'桂' ,'黑' ,'沪','冀','津' ,'京','吉','辽','鲁','蒙',
           '闽','宁','青','琼','陕','苏','晋','皖','湘','新','豫','渝','粤','云','藏','浙')

'''加载数据集 这里定义一个类，有关数据集，测试集的加载'''
class Setloader():
    def __init__(self):
        pass

    def trainset_loader(self):
        path = 'G:\deep_learming\pytorch\data\jk'
        trainset = datasets.ImageFolder(root=path , transform=transform)
        # print(trainset.classes)
        # print(trainset.class_to_idx)
        # print(trainset.imgs)
        # print(trainset[0][1])
        trainloader = torch.utils.data.DataLoader(trainset , batch_size = 16 ,shuffle = True , num_workers = 2)
        return trainloader

    def testset_loader(self):
        path = 'G:\deep_learming\pytorch\data\Test'
        testset =datasets.ImageFolder(root=path, transform=transform)
        testloader = torch.utils.data.DataLoader(testset , batch_size = 12, shuffle =False)
        return testloader


'''定义网络模型的类'''
class Net(nn.Module):  #要继承这个类
    def __init__(self):
        super(Net ,self).__init__() #父类初始化
        '''定义网络结构'''
        self.conv1 = nn.Conv2d(3 , 8 , 3 ) #输入颜色通道 ：1  输出通道：6，卷积核：5*5  卷积核默认步长是1  30*30
        self.conv2 = nn.Conv2d(8 , 16 , 2 )#这是第二个卷积层，输入通道是：6 ，输出通道：16 ，卷积核：3*3   30*30
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层   10*10
        #self.conv3 = nn.Conv2d(16 , 8 , 2)#第三个卷积层，输入层的输入通道要和上一层传下来的通道一样，这里给了填充是2，这个参数默认是：0，填充可以把边缘信息、特征提取出来，不流失   14*14
        self.fc1 = nn.Linear(16*4*4 , 110)
        self.fc2 = nn.Linear(110 , 150)
        self.fc3 = nn.Linear(150 , 130)
        self.fc4 = nn.Linear(130 , 65)        #三个全连接层 65 是最终输出有65个类别

    def forward(self , x):  #前向传播，把整个网络模型，顺序连接起来，init里面只是对层初始化（需要用到的层）
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.conv3(x))
        x = x.view( -1 ,16*4*4)  #转录成可以传入全连接层的的形状
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


'''训练网络模型并保存模型参数的类，训练、测试、显示'''
class Trainandsave():
    def __init__(self):
        self.net=Net()
        pass

    # def use_gpu(self):
    #     if torch.cuda.is_available() ==True:
    #         device = torch.device("cuda:0")
    #         Net.to(device)
    def train_net(self):
        self.net = Net()  # 把模型的定义一下
        criterion = nn.CrossEntropyLoss()  # 定义一个计算损失值 赋给一个对象
        optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(Setloader.trainset_loader(self), 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.net(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    print('[%d ,%5d]loss:%.3f' % (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
        print('finished training!')
        torch.save(self.net.state_dict(), 'zuoye_net_homewords.pkl')  # 训练结束，保存模型参数

    def load_test(self,img):
        self.net.load_state_dict(torch.load('zuoye_net_homewords.pkl'))  #加载模型参数
        Setloader.testset_loader(self)
        datataker = iter(Setloader.testset_loader(self))
        images , labels = datataker.next()
        #print(images)
        # self.imshow(torchvision.utils.make_grid(images))
        # print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(12)))
        # img = cv2.imread('G:\deep_learming\pytorch\data\jk\zh_xiang\debug_char_auxRoi_74.jpg')
        # print(img)
        # img = np.array(img)  #PIL image转换成array
        # print(img)
        img = Image.fromarray(img) #转化成pil格式  array转换成image
        input = transform(img)  #PIL格式的图片可以经过transform装换成torch格式
        # print(input.shape)
        input = input.unsqueeze(0)  #增加一个维度，在原本第0维的位置增加一个维度  才符合pytorch的格式[B,C,H,W]
        # print(input.shape)

        oputs = self.net(input)
        #print(oputs)
        _ ,predicted = torch.max(oputs.data , 1)  # 确定一行中最大的值的索引  torch.max(input, dim)
        # print(predicted)
        print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(1)))
        return classes[predicted]

    def imshow(self ,img):
        img = img/2 +0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg , (1 , 2 ,0)))
        plt.show()
'''数据集分配'''
path = 'G:\研究生课程文件\机器视觉\Ann2'
def set_split(old_path):
    new_path = 'G:\deep_learming\pytorch\data\car_set'
    if os.path.exists(old_path) == 1: #文件夹存在，则新建一个新的文件夹
        os.makedirs(new_path)
    else:
        print('文件夹不存在！')
        return 1
    for path , sub_dirs , files in os.walk(old_path):
        for new_sub_dir in sub_dirs:
            filenames = os.listdir(os.path.join(path,new_sub_dir))   #filmenames 这时就是每个二级文件下 ，每张照片的路径
            filenames = list(filter(lambda x:x.endswith('.jpg') , filenames))   #把flimnames = x ,此时以.png结尾的文件通过过滤器 ，filter语法，后接函数还有序列 第一个为判断函数，第二个为序列
            # print(filenames)
            random.shuffle(filenames) #把序列中所有元素，随机排序

            for i in range(len(filenames)):
                if i < math.floor(0.7 * len(filenames)):#math.floor  向下取整
                    sub_path = os.path.join(new_path , 'Train_set',new_sub_dir)
                elif i <len(filenames):
                    sub_path = os.path.join(new_path , 'Test_set' , new_sub_dir)
                if os.path.exists(sub_path) == 0: #不存在时
                     os.makedirs(sub_path)

                shutil.copy(os.path.join(path, new_sub_dir,filenames[i]) , os.path.join(sub_path , filenames[i]))
class Car_pai():
    def __init__(self):
        img = cv2.imread('G:\deep_learming\pytorch\data\car_pai\jk3.jpg',1)
        imgg= img.copy()

        img = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([120, 220, 255])
        img2 = cv2.inRange(img, lower_blue, upper_blue)  # img1通道是HSV不是BGR了  在指定图像中选定范围像素点
        img2 = cv2.bilateralFilter(img2 ,35 ,75 ,71)
        cv2.namedWindow('show')
        cv2.imshow('show', img2)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        kernel = np.ones((13, 13), np.uint8)
        closing_img = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        closing_img = cv2.bilateralFilter(closing_img,25,75,75)
        contours , hierarchy = cv2.findContours(closing_img,1,2)

        for i in range(len(contours)):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  #左下角 左上角 右上角 右下角
            mianji = rect[1][0] * rect[1][1]  # 车牌面积不应太小 ，太小就丢弃
            # print(mianji)
            if rect[2] >= -45:
                if rect[1][1] != 0:
                    bili = (rect[1][0]) / (rect[1][1])
                else:
                    print('error')
                    bili = 1   #这种情况是不可能的车牌不可能一边是0，这里做特殊处理
            else:
                if rect[1][0] != 0:
                    bili = rect[1][1] / rect[1][0]
            # imgg = cv2.drawContours(imgg, [box], 0, (0, 0, 255), 2)
            if (bili > 2.15 and bili < 4) and mianji>700:
                # print(bili)
                imgg = cv2.drawContours(imgg,[box],0,(0,0,255),2)
                left_point_x = np.min(box[:, 0])
                right_point_x = np.max(box[:, 0])
                top_point_y = np.min(box[:, 1])
                bottom_point_y = np.max(box[:, 1])

                top_left_point = [left_point_x , top_point_y]
                top_right_point = [right_point_x , top_point_y]
                bottom_left_point = [left_point_x , bottom_point_y]
                bottom_right_point = [right_point_x , bottom_point_y]

                vertices = np.array(
                    [top_left_point, top_right_point, bottom_left_point,
                     bottom_right_point])
                # print(vertices)

                pts1 = np.float32(vertices)
                pts2 = np.float32([[0,0],[440,0],[0,140],[440,140]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst= cv2.warpPerspective(imgg , M ,(440,140))
        img2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        img2 = cv2.bilateralFilter(img2,3,75,75)
        ret,img3 = cv2.threshold(img2,130,255,cv2.THRESH_BINARY)
        a = 58
        imgs1= img3[0:140,7:1*a+7]
        imgs2 = img3[0:140,a+7:2*a+7]
        imgs3 = img3[0:140,2*a+28+7:3*a+28+7]
        b = 3*a+28+5
        imgs4 = img3[0:140,b:b+a]
        imgs5 = img3[0:140,b+a:b+2*a]
        imgs6 = img3[0:140,b+2*a:b+3*a]
        imgs7 = img3[0:140,b+3*a:b+4*a]
        imges = [imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, imgs7]
        str1 = ['', '', '', '', '', '', '']
        for i in range(len(imges)):
            imges[i]=cv2.cvtColor(imges[i],cv2.COLOR_GRAY2BGR)
            imges[i]=cv2.medianBlur(imges[i],5)
            imges[i]=cv2.resize(imges[i],(20,20))
            imges[i] = cv2.bilateralFilter(imges[i],55,55,75)
            # cv2.imwrite('G:\deep_learming\pytorch\data\%d.jpg'%i , imges[i])
            train = Trainandsave()
            str1[i]=train.load_test(imges[i])
            # print(self.str1[i])



        for i in range(len(imges)):
            plt.subplot(1,8,i+1),plt.imshow(imges[i])
            plt.xticks([]),plt.yticks([])

        plt.show()
        result = str1[0]+str1[1]+str1[2]+str1[3]+str1[4]+str1[5]+str1[6]
        print('结果是：'+result)
        # imghj = cv2ImgAddText(imgg,result,top_left_point[0],top_left_point[1]-23)

        cv2.namedWindow('show')
        cv2.imshow('show',imgg)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()

def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=25):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    # set_split(path)
    # trainloader=Setloader()
    # trainloader = trainloader.trainset_loader()
    shishi = Car_pai()  #车牌识别定位分割
    # img = cv2.imread('G:\deep_learming\pytorch\data\jk16.jpg',1)
    #
    # train = Trainandsave()
    # train.train_net()  #训练网络，保存参数
   # train.load_test(img)
    #


