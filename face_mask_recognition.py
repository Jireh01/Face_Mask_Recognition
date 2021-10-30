import cv2
import os
import mediapipe as mp
deteccion_cara = mp.solutions.face_detection
etiqeta = ["Con_mascarilla", "Sin_mascarilla"]
# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("train_model.xml")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with deteccion_cara.FaceDetection(
     min_detection_confidence=0.5) as face_detection:
     while True:
          ret, frame = cap.read()
          if ret == False: break
          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_detection.process(frame_rgb)
          if results.detections is not None:
               for detection in results.detections:
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    if xmin < 0 and ymin < 0:
                         continue
                    frame_size = frame[ymin : ymin + h, xmin : xmin + w]
                    frame_size = cv2.cvtColor(frame_size, cv2.COLOR_BGR2GRAY)
                    frame_size = cv2.resize(frame_size, (72, 72), interpolation=cv2.INTER_CUBIC)
                    
                    result = face_mask.predict(frame_size)
                    if result[1] < 150:
                         color = (0, 255, 0) if etiqeta[result[0]] == "Con_mascarilla" else (0, 0, 255)
                         cv2.putText(frame, "{}".format(etiqeta[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                         cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
          cv2.imshow("Frame", frame)
          k = cv2.waitKey(1)
          if k == 27:
               break
cap.release()
cv2.destroyAllWindows()