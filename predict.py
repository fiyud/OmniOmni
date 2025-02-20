from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("D:/Eggg/CVR EGG 4.v2i.yolov11/weight16/weights/best.pt")

    model.predict(source=r"D:\Eggg\bky", save=True, imgsz=640, device=0)