####IVC ROBOTICS TEAM###
#Prototype - Yolo implementation

from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator


model = YOLO('best.pt') #bottle model
cap = cv2.VideoCapture(0) #input from camera
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img, conf=0.6) #here adjust confidence

    for r in results:
        
        annotator = Annotator(frame)

        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            #annotator.box_label(b, model.names[int(c)])
            annotator.box_label(b, str(b))

            x1 = b.numpy()[0]
            y1 = b.numpy()[1]
            x2 = b.numpy()[2]
            y2 = b.numpy()[3]
            print("Coordinates: ", x1, y1, x2, y2)
            #print("Datatype of b is: ", type(b))

            circle1 = cv2.circle(frame, (int(x1+((x2-x1)/2)),int(y1+((y2-y1)/2)) ), 6, (0, 255, 100), 4)

            print(b)

    frame = annotator.result()

    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
