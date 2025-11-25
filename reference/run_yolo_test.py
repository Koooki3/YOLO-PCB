import cv2
import numpy as np
import os

model_path = r'D:\VisualRobot-Git\VisualRobot\models\best.onnx'
image_path = r'Img/defect2_041.jpg'
output_path = r'Img/yolo_test_out.jpg'

if not os.path.exists(model_path):
    print('Model not found:', model_path)
    raise SystemExit(1)
if not os.path.exists(image_path):
    print('Image not found:', image_path)
    raise SystemExit(1)

net = cv2.dnn.readNet(model_path)
print('Loaded model')

img = cv2.imread(image_path)
if img is None:
    print('Failed to read image', image_path)
    raise SystemExit(1)

inp_size = (640,640)
blob = cv2.dnn.blobFromImage(img, 1.0/255.0, inp_size, (0,0,0), swapRB=True, crop=False)
net.setInput(blob)

outs = net.forward(net.getUnconnectedOutLayersNames())
print('Number of output blobs:', len(outs))
for i,o in enumerate(outs):
    print(i, 'shape', np.array(o).shape)

# Merge outputs if multiple
if len(outs) == 1:
    dets = np.array(outs[0])
else:
    dets = np.vstack([np.array(x).reshape(-1, np.array(x).shape[-1]) for x in outs])

print('Merged detections shape:', dets.shape)

# Normalize possible shapes
if dets.ndim == 3 and dets.shape[0] == 1:
    dets = dets[0]

# Heuristic: expect NxC where C>=6
h, w = img.shape[:2]
if dets.shape[1] < 6:
    print('Unexpected output columns:', dets.shape)
else:
    nc = dets.shape[1] - 5
    print('Detected rows:', dets.shape[0], 'num_classes:', nc)

    boxes = []
    scores = []
    classIds = []
    confThresh = 0.25
    nmsThresh = 0.45

    for i in range(dets.shape[0]):
        row = dets[i]
        cx, cy, bw, bh, obj = row[0], row[1], row[2], row[3], row[4]
        if nc <= 0:
            continue
        class_scores = row[5:5+nc]
        cls = int(np.argmax(class_scores))
        cls_score = class_scores[cls]
        conf = float(obj * cls_score)
        if conf < confThresh:
            continue
        # xywh -> convert to pixels
        x1 = int((cx - bw/2.0) * (w/ inp_size[0]))
        y1 = int((cy - bh/2.0) * (h/ inp_size[1]))
        boxw = int(bw * (w/ inp_size[0]))
        boxh = int(bh * (h/ inp_size[1]))
        boxes.append([x1, y1, boxw, boxh])
        scores.append(conf)
        classIds.append(cls)

    # NMS
    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, confThresh, nmsThresh)
        for idx in idxs:
            i = idx[0] if isinstance(idx, (list,tuple,np.ndarray)) else int(idx)
            x,y,ww,hh = boxes[i]
            cls = classIds[i]
            conf = scores[i]
            color = (0,255,0) if cls==0 else (0,0,255)
            cv2.rectangle(img, (x,y), (x+ww, y+hh), color, 2)
            label = f'{cls}:{conf:.2f}'
            cv2.putText(img, label, (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

    cv2.imwrite(output_path, img)
    print('Saved result to', output_path)

print('Done')
