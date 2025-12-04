# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:27:20 2024

@author: AM4
"""
# Импортируем библиотеки
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ultralytics
from ultralytics import YOLO
import cv2
#import matplotlib.pyplot as plt

# Проверяем что доступно из оборудования
ultralytics.checks()



# Теперь попробуем обучить на собственном датасете
# он доступен по ссылке https://drive.google.com/file/d/1qS_yGj3vkmEuv9Fc3Xffqg6fE5C8m3AG/view?usp=drive_link

# перед обучением необходимо скорректировать пути в файле masked.yaml
# пути должны быть абсолютными
# обучение может занять много времени, особенно на CPU
model = YOLO("yolov8s.pt")
results = model.train(data="raccoon.yaml", model="yolov8s.pt", epochs=2, batch=6,
                      project='raccoon_detection', val = True, verbose=True)

#results = model("rac1.jpg")
results = model("maxresdefault.jpg")
# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


# Попробуем обработать видео

# Открываем видеофайл
video_path = "masktrack.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    # Считываем кадр
    success, frame = cap.read()

    if success:
        # Если кадр прочитался успешно, запускаем модель
        results = model(frame)
        
        result = results[0]
        cv2.imshow("YOLOv8", result.plot())

        # По нажатию "q" будем выходить из цикла
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Освобождаем поток видео и закрываем окно отображения
cap.release()
cv2.destroyAllWindows()