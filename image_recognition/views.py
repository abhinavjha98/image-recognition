from django.shortcuts import render
import os
import cv2
import torch
import time
import traceback
from .models import *
model = torch.hub.load('./static/yolov5/', 'custom', path='./static/best.pt',
                       source='local')

faceProto = "./static/deploy.prototxt"
faceModel = "./static/res10_300x300_ssd_iter_140000_fp16.caffemodel"
genderProto = "./static/gender_deploy.prototxt"
genderModel = "./static/gender_net.caffemodel"
ageProto = "./static/age_deploy.prototxt"
ageModel = "./static/age_net.caffemodel"
genderList = ['Male', 'Female']
ageList = ['(0-3)', '(4-7)', '(8-14)', '(15-25)',
           '(26-42)', '(43-50)', '(51-59)', '(60-100)']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderProto, genderModel)
ageNet = cv2.dnn.readNet(ageProto, ageModel)

def highlightFace(net, frameOpencvDnn):
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])       
    return faceBoxes

def handle_uploaded_file(f):
    global uploaded_image
    f_u=f.name
    uploaded_image1=str('{}_{}.{}'.format(f_u.split('.')[0],time.time(),f_u.split('.')[-1]))
    uploaded_image=uploaded_image1
    with open('./media/{}'.format(uploaded_image1), 'wb') as destination:
        for chunk in f.chunks():
            destination.write(chunk) 

def homepage(request):
    try:
        start_time=time.time()
        x=1
        file_name=''
        email=''
        url1=''
        url2=''
        img=''
        gen=[0,0]
        age_no=[0,0,0,0,0,0,0,0]
        if request.method=="POST":
            file_name=request.FILES['file']
            handle_uploaded_file(file_name)
            email=request.POST['email']
            x=2
        if x==2:
            img = cv2.imread('./media/{}'.format(uploaded_image))
            img_crop = cv2.imread('./media/{}'.format(uploaded_image))
            h, w, c = img.shape
            if h > 3000:
                img = cv2.resize(img, (int((w*50)/100), int((h*50)/100)))
            results = model(img, size=600)
            results = results.pandas().xyxy[0].to_dict(orient="records")
            count = 0
            padding = 20
            for result in results:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                if cs == 0 and con>0.35:
                    count += 1
                    imgx = img_crop[y1:y2, x1:x2]
                    #if os.path.isfile('/media/'+'{}'.format(uploaded_image))==False:
                    #   continue
                    if h > 1000:
                        text = "Person{} {}".format(count, int(con*100))
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0),5)
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 5)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+30), (x1+text_w, y1+30+text_h+4), (255, 0, 0), -1)
                        cv2.putText(img, text, (x1, y1 + 30+text_h+4),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 255), 5)
                    else:
                        text = "Person{} {}".format(count, int(con*100))
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0),2)
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+30), (x1+text_w, y1+30+text_h+4), (255, 0, 0), -1)
                        cv2.putText(img, text, (x1, y1 + 30+text_h+4),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                    gender = ''
                    age = ''
                    faceBoxes = highlightFace(faceNet, imgx)
                    for faceBox in faceBoxes:
                        face = imgx[max(0, faceBox[1]-padding):
                                    min(faceBox[3]+padding, imgx.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, imgx.shape[1]-1)]
                        if len(face)==0:
                            continue
                        blob = cv2.dnn.blobFromImage(
                            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        age = ageList[agePreds[0].argmax()]
                        break
                    if age == ''or gender == '':
                        continue
                    if gender=='Male':
                        gen[0]+=1
                    elif gender=='Female':
                        gen[1]+=1

                    if age=='(0-3)':
                        age_no[0]+=1
                    elif age=='(4-7)':
                        age_no[1]+=1
                    elif age=='(8-14)':
                        age_no[2]+=1
                    elif age=='(15-25)':
                        age_no[3]+=1
                    elif age=='(26-42)':
                        age_no[4]+=1
                    elif age=='(43-50)':
                        age_no[5]+=1
                    elif age=='(51-59)':
                        age_no[6]+=1
                    elif age=='(60-100)':
                        age_no[7]+=1


                    '''if h > 1000:
                        text_size, _ = cv2.getTextSize(
                            gender, cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 5)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+120), (x1+text_w, y1+120+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, gender, (x1, y1 + 120+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 255), 5)
                        text_size, _ = cv2.getTextSize(
                            age+"years", cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 5)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+210), (x1+text_w, y1+210+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, age+"years", (x1, y1 + 210+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 255), 5)
                    else:
                        text_size, _ = cv2.getTextSize(
                            gender, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+60), (x1+text_w, y1+60+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, gender, (x1, y1 + 60+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                        text_size, _ = cv2.getTextSize(
                            age+"years", cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+90), (x1+text_w, y1+90+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, age+"years", (x1, y1 + 90+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)'''
            '''if h > 1000:
                cv2.putText(img, "Number of person in image : {}".format(
                    count), (20, h-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 0), 5)
                
            else:
                cv2.putText(img, "Number of person in image : {}".format(
                    count), (20, h-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1)'''
            if w<h:
                img=cv2.resize(img,(600,800))
            else: 
                img=cv2.resize(img,(800,600))
            current_time=time.time()
            cv2.imwrite("./predicted_image_{}.{}".format(current_time,file_name.name.split('.')[-1]), img)
            cv2.imwrite("./media/predicted_image_{}.{}".format(current_time,file_name.name.split('.')[-1]), img)
            url1 = "./predicted_image_{}.{}".format(current_time,file_name.name.split('.')[-1])
            url2 = "./media/predicted_image_{}.{}".format(current_time,file_name.name.split('.')[-1])
            data=img_recog_table(uploaded_Img=file_name,email=email,processed_image=url1)
            data.save() 
            end_time=time.time()
            print("Total time taken:",end_time-start_time)
        if x==1:
            return render(request,'index.html')
        else:
            return render(request,'sample.html',{'x':0,'pred_img':url2,'person_count':count,'age':age_no,'gender':gen})
    except:
        traceback.print_exc()
        
    

def img_recognition(request):
    try:
        start_time=time.time()
        x=1
        file_name=''
        email=''
        url1=''
        url2=''
        img=''
        gen=[0,0]
        age_no=[0,0,0,0,0,0,0,0]
        if request.method=="POST":
            file_name=request.FILES['file']
            handle_uploaded_file(file_name)
            email=request.POST['email']
            x=2
        if x==2:
            img = cv2.imread('./media/'+'{}'.format(uploaded_image))
            img_crop = cv2.imread('./media/'+'{}'.format(uploaded_image))
            h, w, c = img.shape
            if h > 3000:
                img = cv2.resize(img, (int((w*50)/100), int((h*50)/100)))
            results = model(img, size=600)
            results = results.pandas().xyxy[0].to_dict(orient="records")
            count = 0 
            padding = 20
            for result in results:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                if cs == 0 and con>0.35:
                    count += 1
                    imgx = img_crop[y1:y2, x1:x2]
                    #if os.path.isfile('./media/'+'{}'.format(uploaded_image))==False:
                    #    continue
                    if h > 1000:
                        text = "Person{} {}".format(count, int(con*100))
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0),5)
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 5)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+30), (x1+text_w, y1+30+text_h+4), (255, 0, 0), -1)
                        cv2.putText(img, text, (x1, y1 + 30+text_h+4),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 255), 5)
                    else:
                        text = "Person{} {}".format(count, int(con*100))
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0),2)
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+30), (x1+text_w, y1+30+text_h+4), (255, 0, 0), -1)
                        cv2.putText(img, text, (x1, y1 + 30+text_h+4),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                    gender = ''
                    age = ''
                    faceBoxes = highlightFace(faceNet, imgx)
                    for faceBox in faceBoxes:
                        face = imgx[max(0, faceBox[1]-padding):
                                    min(faceBox[3]+padding, imgx.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, imgx.shape[1]-1)]
                        if len(face)==0:
                            continue
                        blob = cv2.dnn.blobFromImage(
                            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        age = ageList[agePreds[0].argmax()]
                        break
                    if age == ''or gender == '':
                        continue
                    if gender=='Male':
                        gen[0]+=1
                    elif gender=='Female':
                        gen[1]+=1

                    if age=='(0-3)':
                        age_no[0]+=1
                    elif age=='(4-7)':
                        age_no[1]+=1
                    elif age=='(8-14)':
                        age_no[2]+=1
                    elif age=='(15-25)':
                        age_no[3]+=1
                    elif age=='(26-42)':
                        age_no[4]+=1
                    elif age=='(43-50)':
                        age_no[5]+=1
                    elif age=='(51-59)':
                        age_no[6]+=1
                    elif age=='(60-100)':
                        age_no[7]+=1


                    '''if h > 1000:
                        text_size, _ = cv2.getTextSize(
                            gender, cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 5)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+120), (x1+text_w, y1+120+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, gender, (x1, y1 + 120+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 255), 5)
                        text_size, _ = cv2.getTextSize(
                            age+"years", cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, 5)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+210), (x1+text_w, y1+210+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, age+"years", (x1, y1 + 210+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 255), 5)
                    else:
                        text_size, _ = cv2.getTextSize(
                            gender, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+60), (x1+text_w, y1+60+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, gender, (x1, y1 + 60+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                        text_size, _ = cv2.getTextSize(
                            age+"years", cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        text_w, text_h = text_size
                        img = cv2.rectangle(
                            img, (x1, y1+90), (x1+text_w, y1+90+text_h+2), (255, 0, 0), -1)
                        cv2.putText(img, age+"years", (x1, y1 + 90+text_h+1),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)'''
            '''if h > 1000:
                cv2.putText(img, "Number of person in image : {}".format(
                    count), (20, h-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 255, 0), 5)
                
            else:
                cv2.putText(img, "Number of person in image : {}".format(
                    count), (20, h-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1)'''
            if w<h:
                img=cv2.resize(img,(600,800))
            else: 
                img=cv2.resize(img,(800,600))
            curr_time=time.time()
            cv2.imwrite("./predicted_image_{}.{}".format(curr_time,file_name.name.split('.')[-1]), img)
            cv2.imwrite("./media/predicted_image_{}.{}".format(curr_time,file_name.name.split('.')[-1]), img)
            url1 = "./predicted_image_{}.{}".format(curr_time,file_name.name.split('.')[-1])
            url2 = "./media/predicted_image_{}.{}".format(curr_time,file_name.name.split('.')[-1])
            data=img_recog_table(uploaded_Img=file_name,email=email,processed_image=url1)
            data.save() 
            end_time=time.time()
            print("Total time taken:",end_time-start_time)
        if x==1:
            return render(request,'sample.html',{'x':1,'person_count':3,'age':[0,0,0,1,2,0,0,0],'gender':[2,1]})
        else:
            return render(request,'sample.html',{'x':0,'pred_img':url2,'person_count':count,'age':age_no,'gender':gen})
    except:
        traceback.print_exc()
        