import  cv2
import  dlib
import math

cap=cv2.VideoCapture(0)



glass=cv2.imread("mask4.png")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def gradient(pt1,pt2):
    if (pt2[0]-pt1[0])==0:
        return ("a")
    else:
        return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

def getangle(points):
    pt1,pt2,pt3=points[-3:]
    if gradient(pt1,pt2) =="a":
        pass
    else:
        m1=gradient(pt1,pt2)
    if gradient(pt1, pt3) =="a":
        pass
    else:
        m2=gradient(pt1,pt3)
    angR=math.atan((m2-m1)/(1+(m2*m1)))
    angD=round(math.degrees(angR))
    print(angD)
points=[]
while True:
    success,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)

    for face in faces:
        landmark=predictor(gray,face)

        top_head=(landmark.part(30).x,landmark.part(30).y)
        left_eye=(landmark.part(27).x,landmark.part(27).y)
        right_eye=(landmark.part(8).x,landmark.part(8).y)
        face1=(landmark.part(48).x,landmark.part(48).y)
        face2 = (landmark.part(54).x, landmark.part(54).y)

        eye_width=int(math.hypot(left_eye[0]-right_eye[0],left_eye[1]-right_eye[1])*1.2)
        eye_hight=int(eye_width*1.4)

        '''cv2.circle(img,top_head,3,(0,255,0),-1)
        #cv2.circle(img, left_eye, 3, (0, 255, 0), -1)
        cv2.circle(img, right_eye, 3, (0, 255, 0), -1)
        cv2.circle(img,(right_eye[0]-50,right_eye[1]), 3, (0, 255, 0), -1)'''

        smile = int(math.hypot(face1[0] - face2[0], face1[1] - face2[1]))
        #print(smile)
        points = ((right_eye[0] - 50, right_eye[1]), top_head, right_eye)
        #ag = getangle(points)
        top_left=(int(top_head[0]-eye_width/2),
                           int(top_head[1]-eye_hight/2))
        bottom_right=(int(top_head[0]+eye_width/2),
                       int(top_head[1]+eye_hight/2))

        '''cv2.rectangle(img,(int(top_head[0]-eye_width/2),
                           int(top_head[1]-eye_hight/2)),
                      (int(top_head[0]+eye_width/2),
                       int(top_head[1]+eye_hight/2)),(0,255,0),2)'''
                       
        if smile>64:
            eye_resize = cv2.resize(glass, (eye_width, eye_hight))
            eye_area=(img[top_left[1]:top_left[1] +eye_hight,top_left[0]:top_left[0]+eye_width])

            eye_gray=cv2.cvtColor(eye_resize,cv2.COLOR_BGR2GRAY)
            _,eye_mask=cv2.threshold(eye_gray,25,255,cv2.THRESH_BINARY_INV)

            new_eye=cv2.bitwise_and(eye_area,eye_area, mask=eye_mask)
            #print(new_eye)
            final=cv2.add(new_eye,eye_resize)

            img[top_left[1]:top_left[1] + eye_hight,
            top_left[0]:top_left[0] + eye_width]=final


            #cv2.imshow("final", final)


            '''cv2.imshow("Video", img)
            cv2.imshow('sunglass pic', eye_resize)
            cv2.imshow("area", eye_area)
    
            cv2.imshow("mask",eye_mask )'''

            cv2.imshow("Video", img)
        else:
            cv2.imshow("Video", img)

    if cv2.waitKey(1)  & 0xFF ==ord("q"):
        break
