from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator


model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(frame)

        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            #annotator.box_label(b, model.names[int(c)])
            annotator.box_label(b, str(b))

            x = b.numpy()[0]
            y = b.numpy()[1]
            h = b.numpy()[2]
            w = b.numpy()[3]
            print("Coordinates: ", x, y, w, h)
            #print("Datatype of b is: ", type(b))

            #circle1 = cv2.circle(frame, (int(0.492 + 0.469), int(0.625 + 0.416)), 30, (0, 255, 100), 2)
            circle1 = cv2.circle(frame, (int(x+(h/4)), int(y+(h/2))), 30, (0, 255, 100), 2)

            print(b)

    frame = annotator.result()
    #circle1 = cv2.circle(frame, (320, 240), 30, (0, 255, 100), 2)
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
