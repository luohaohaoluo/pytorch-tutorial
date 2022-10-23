import time
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from models import fasterrcnn_fpn_resnet50

labels_map = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}


def test():
    labels_temp = {str(value): key for key, value in labels_map.items()}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 21

    model = fasterrcnn_fpn_resnet50(num_classes)
    model.load_state_dict(torch.load("./model.pth"))

    # model = torch.load("./model_all.pth")
    model.to(device)

    model.eval()

    transform = torchvision.transforms.ToTensor()

    # 调用摄像头
    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("test.mp4")

    # fps = 0.0
    model.eval()
    with torch.no_grad():
        while True:
            # t1 = time.time()
            ret, frame = capture.read()
            if not ret:
                print("识别完成！")
                break

            image = frame
            frame = Image.fromarray(frame)
            # print(type(frame))
            frame = transform(frame)
            # fps = (fps + (1. / (time.time() - t1))) / 2
            prediction = model([frame.to(device)])

            boxes = prediction[0]["boxes"].cpu().numpy()
            scores = prediction[0]["scores"].cpu().numpy()
            labels = prediction[0]["labels"].cpu().numpy()
            step = 0
            for score in scores:
                if score > 0.50:
                    step += 1

            boxes = boxes[0:step, :]
            labels = labels[0:step]
            scores = scores[0:step]
            real_label = []

            for i in labels:
                real_label.append(labels_temp.get(str(i)))

            for index in range(step):
                box = boxes[index]
                cv2.rectangle(img=image, pt1=[int(box[0]), int(box[1])], pt2=[int(box[2]), int(box[3])],
                              color=(0, 0, 225), thickness=3)
                texts = real_label[index] + ":" + str(np.round(scores[index], 2))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, texts, (int(box[0]), int(box[1])), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
                # cv2.putText(image, f"fps = {fps:.2f}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.namedWindow("Object Classification", 0)
            cv2.resizeWindow("Object Classification", 1024, 600)
            cv2.imshow('Object Classification', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
