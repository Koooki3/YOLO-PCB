import onnxruntime as ort
import numpy as np
import cv2
import os

model_path = r'D:\VisualRobot-Git\VisualRobot\models\best.onnx'
image_path = r'Img/defect2_041.jpg'
output_path = r'Img/yolo_test_ort_out.jpg'

if not os.path.exists(model_path):
    print('Model not found:', model_path)
    raise SystemExit(1)
if not os.path.exists(image_path):
    print('Image not found:', image_path)
    raise SystemExit(1)

print('Creating ONNX Runtime session...')
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
try:
    sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
except Exception as e:
    print('Failed to create ORT session:', e)
    raise

print('Session created. Inputs:')
for inp in sess.get_inputs():
    print(' ', inp.name, inp.shape, inp.type)
for out in sess.get_outputs():
    print(' Output:', out.name, out.shape, out.type)

img = cv2.imread(image_path)
h, w = img.shape[:2]
input_size = (640, 640)

blob = cv2.resize(img, input_size)
blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
blob = blob.astype('float32') / 255.0
blob = np.transpose(blob, (2,0,1))
blob = np.expand_dims(blob, 0)

input_name = sess.get_inputs()[0].name
print('Running inference...')
outputs = sess.run(None, {input_name: blob})
print('Got', len(outputs), 'output(s)')

# Normalize outputs to NxC matrix if possible
outs = outputs
if len(outs) == 1:
    dets = np.array(outs[0])
else:
    # try to concat if shapes allow
    try:
        dets = np.vstack([np.array(x).reshape(-1, np.array(x).shape[-1]) for x in outs])
    except Exception:
        dets = np.array(outs[0])

if dets.ndim == 3 and dets.shape[0] == 1:
    dets = dets[0]

print('Detections array shape:', dets.shape)

if dets.ndim == 2 and dets.shape[1] >= 6:
    nc = dets.shape[1] - 5
    conf_thresh = 0.25
    nms_thresh = 0.45
    boxes = []
    scores = []
    classIds = []
    for i in range(dets.shape[0]):
        row = dets[i]
        cx, cy, bw, bh, obj = row[0], row[1], row[2], row[3], row[4]
        class_scores = row[5:5+nc]
        if class_scores.size == 0:
            continue
        cls = int(np.argmax(class_scores))
        cls_score = float(class_scores[cls])
        conf = float(obj) * cls_score
        if conf < conf_thresh:
            continue
        # map xywh (relative to input_size) -> pixel coords in original image
        scale_x = w / input_size[0]
        scale_y = h / input_size[1]
        x1 = int((cx - bw/2.0) * scale_x)
        y1 = int((cy - bh/2.0) * scale_y)
        boxw = int(bw * scale_x)
        boxh = int(bh * scale_y)
        boxes.append([x1, y1, boxw, boxh])
        scores.append(conf)
        classIds.append(cls)

    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
        for idx in idxs:
            i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
            x,y,ww,hh = boxes[i]
            cls = classIds[i]
            conf = scores[i]
            color = (0,255,0) if cls==0 else (0,0,255)
            cv2.rectangle(img, (x,y), (x+ww, y+hh), color, 2)
            label = f'{cls}:{conf:.2f}'
            cv2.putText(img, label, (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

    cv2.imwrite(output_path, img)
    print('Saved result to', output_path)
else:
    print('Output format not recognized for simple postprocess. Inspect outputs manually.')

print('Done')
