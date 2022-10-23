import cv2
import numpy as np
import random
import torch
import argparse

from PIL import Image
from torchvision.transforms import transforms as transforms
from models import maskrcnn_fpn_resnet50

lable_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
               "dog", "horse", "motorbike", "person", "potted plant",
               "sheep", "sofa", "train", "tv/monitor"]

COLORS = np.random.uniform(0, 255, size=(len(lable_names), 3))


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [lable_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels


def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)

    return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='path to the input data')
    parser.add_argument('-t', '--threshold', default=0.6, type=float,
                        help='score threshold for discarding detection')
    args = vars(parser.parse_args(args=['-i', 'test1.jpg']))

    num_classes = 21
    # initialize the model
    model = maskrcnn_fpn_resnet50(num_classes)
    model.load_state_dict(torch.load("./model.pth"))
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()
    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_path = args['input']
    image = Image.open(image_path).convert('RGB')
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    # transform the image
    image = transform(image)
    # print(image)

    # add a batch dimension

    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, model, args['threshold'])
    result = draw_segmentation_map(orig_image, masks, boxes, labels)

    # visualize the image
    cv2.imshow('Segmented image', np.array(result))
    cv2.waitKey(0)

    # set the save path
    # save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}.jpg"
    # cv2.imwrite(save_path, result)
