import cv2
import numpy as np

def emptyFunction():
    pass

def main():
    windowName = "Result"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(3)
       
    lowerBound_B = np.array([100,50,50])
    upperBound_B = np.array([140,255,255])
    
    lowerBound_G = np.array([33,80,40])
    upperBound_G = np.array([102,255,170])
    
    lowerBound_R = np.array([155,175,185])
    upperBound_R = np.array([185,220,285])
        
    lowerBound_Y = np.array([20,186,215])
    upperBound_Y = np.array([40,206,295])
    
    kernelOpen = np.ones((5,5))
    kernelClosed = np.ones((20,20))
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
        
    while(ret):
        ret, frame = cap.read()
        
        original = frame
                
        #converting BGR to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #mask for blue color
        mask_B = cv2.inRange(frame_hsv, lowerBound_B, upperBound_B)
        maskOpen_B = cv2.morphologyEx(mask_B,cv2.MORPH_OPEN, kernelOpen)
        maskClose_B =cv2.morphologyEx(maskOpen_B, cv2.MORPH_CLOSE, kernelClosed)
        maskFinal_B = maskClose_B
        frame1, contours, hierarchy = cv2.findContours(maskFinal_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(original, [approx], -1, (255, 255, 255), 2)
        for c in contours[:]:
            M = cv2.moments(c)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.circle(original, (cX, cY), 2, (255,255,255), -1)
            cv2.putText(original, "blue", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        #mask for green color
        mask_G = cv2.inRange(frame_hsv, lowerBound_G, upperBound_G)
        maskOpen_G = cv2.morphologyEx(mask_G,cv2.MORPH_OPEN, kernelOpen)
        maskClose_G =cv2.morphologyEx(maskOpen_G, cv2.MORPH_CLOSE, kernelClosed)
        maskFinal_G = maskClose_G
        frame2, contours, hierarchy = cv2.findContours(maskFinal_G, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            epsilon = 0.005 * cv2.arcLength(c, True) #for precision
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(original, [approx], -1, (255,255,255), 2)        
        for c in contours[:]:
            M = cv2.moments(c)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.circle(original, (cX, cY), 2, (255,255,255), -1)
            cv2.putText(original, "green", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        #mask for red color
        mask_R = cv2.inRange(frame_hsv, lowerBound_R, upperBound_R)
        maskOpen_R = cv2.morphologyEx(mask_R,cv2.MORPH_OPEN, kernelOpen)
        maskClose_R =cv2.morphologyEx(maskOpen_R, cv2.MORPH_CLOSE, kernelClosed)
        maskFinal_R = maskClose_R
        frame3, contours, hierarchy = cv2.findContours(maskFinal_R, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(original, [approx], -1, (255,255,255), 2)
        for c in contours[:]:
            M = cv2.moments(c)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.circle(original, (cX, cY), 2, (255,255,255), -1)
            cv2.putText(original, "red", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
        #mask for yellow color
        mask_Y = cv2.inRange(frame_hsv, lowerBound_Y, upperBound_Y)
        maskOpen_Y = cv2.morphologyEx(mask_Y,cv2.MORPH_OPEN, kernelOpen)
        maskClose_Y =cv2.morphologyEx(maskOpen_Y, cv2.MORPH_CLOSE, kernelClosed)
        maskFinal_Y = maskClose_Y
        frame4, contours, hierarchy = cv2.findContours(maskFinal_Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(original, [approx], -1, (255,255,255), 2)
        for c in contours[:]:
            M = cv2.moments(c)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.circle(original, (cX, cY), 2, (255,255,255), -1)
            cv2.putText(original, "yellow", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        
        cv2.imshow(windowName, original)
        
        if cv2.waitKey(1) == 27:
            break
                
    cv2.destroyAllWindows()
    cap.release()
    
if __name__ == "__main__":
    main()