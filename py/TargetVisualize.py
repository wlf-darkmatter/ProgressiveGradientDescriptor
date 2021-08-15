
from xml.dom.minidom import parse
import xml.dom.minidom
import sys
import cv2,os
import string


def readxml():


    domTree=parse("D:\\data\\FAIR1M\\train\\part1\\labelXmls\\0.xml")
    
    rootNode=domTree.documentElement
    
    #print(rootNode.nodeName)

    objects=rootNode.getElementsByTagName("objects")
    #points=object.getElementsByTagName("points")
    for object in rootNode.getElementsByTagName('object'):
        print('1')

        #ty1=object.getElementsByTagName('type')[0]
        #print(ty1.childNodes[0].data)

        point0=object.getElementsByTagName('point')[0]
        #print(point0.childNodes[0].data)
        position0=point0.childNodes[0].data
        position0=position0.split(',')
        print(position0)
        x0=int(float(position0[0]));
        y0=int(float(position0[1]));
        print(x0)
        print(y0)

        point1=object.getElementsByTagName('point')[1]
        #print(point1.childNodes[0].data)
        position1=point1.childNodes[0].data
        position1=position1.split(',')
        print(position1)
        x1=int(float(position1[0]))
        y1=int(float(position1[1]))
        print(x1)
        print(y1)

        point2=object.getElementsByTagName('point')[2]
        #print(point1.childNodes[0].data)
        position2=point2.childNodes[0].data
        position2=position2.split(',')
        print(position2)
        x2=int(float(position2[0]))
        y2=int(float(position2[1]))
        print(x2)
        print(y2)

        point3=object.getElementsByTagName('point')[3]
        #print(point1.childNodes[0].data)
        position3=point3.childNodes[0].data
        position3=position3.split(',')
        print(position3)
        x3=int(float(position3[0]))
        y3=int(float(position3[1]))
        print(x3)
        print(y3)

        point4=object.getElementsByTagName('point')[4]
        #print(point1.childNodes[0].data)
        position4=point4.childNodes[0].data
        position4=position4.split(',')
        print(position4)
        x4=int(float(position4[0]))
        y4=int(float(position4[1]))
        print(x4)
        print(y4)

        #######划线
        ptStart=(x0,y0)
        ptEnd=(x1,y1)
        print(type(x0))
        point_Color=(0,255,0)
        thickness=1
        linetype=4

        cv2.line(img,ptStart,ptEnd,point_Color,thickness,linetype)

        ptStart=(x1,y1)
        ptEnd=(x2,y2)
        cv2.line(img,ptStart,ptEnd,point_Color,thickness,linetype)

        ptStart=(x2,y2)
        ptEnd=(x3,y3)
        cv2.line(img,ptStart,ptEnd,point_Color,thickness,linetype)

        ptStart=(x3,y3)
        ptEnd=(x4,y4)
        
        cv2.line(img,ptStart,ptEnd,point_Color,thickness,linetype)






img=cv2.imread("D:\\data\\FAIR1M\\train\\part1\\images\\0.tif")
readxml()
cv2.imshow('img',img)
cv2.imwrite("D:\\data\\data1\\0.tif",img)
cv2.waitKey(0)


