import cv2
import setproctitle as setproctitle
import onnxruntime
import time
from utils.utils import *
WEIGHTS_PATH1 = 'cuda_onnx/yolov5s.onnx'
WEIGHTS_PATH2 = 'cuda_onnx/yolov5x.onnx'

CUDA = False
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if CUDA else ['CPUExecutionProvider']


session1 = onnxruntime.InferenceSession(WEIGHTS_PATH1, providers=providers)
session2 = onnxruntime.InferenceSession(WEIGHTS_PATH2, providers=providers)
output_names = [x.name for x in session1.get_outputs()]
meta1 = session1.get_modelmeta().custom_metadata_map  # metadata

if 'stride' in meta1:
    stride1, names1 = int(meta1['stride']), eval(meta1['names'])

meta2 = session2.get_modelmeta().custom_metadata_map  # metadata

if 'stride' in meta2:
    stride2, names2 = int(meta2['stride']), eval(meta2['names'])


fp16 = False
img_size = 640
nhwc = True
frameIndex = 0
stride1 = 32
auto = False
w = 1080
h = 1920

wh = w * h
hwhh = int(wh / 4)
dh = h + int(h / 2)
SHAPE = [1088, 1920]
ACQUISITIONSIZE_Y = (3, SHAPE[0] * SHAPE[1])
ACQUISITIONSIZE_U = (3, ACQUISITIONSIZE_Y[1] // 4)
ACQUISITIONSIZE_V = (3, ACQUISITIONSIZE_U[1])


def run(framesQueue, predictionQueue):
    setproctitle.setproctitle("Inference")

    while True:

        try:
            element = framesQueue.get()
            im = element.copy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t_inf = time.time()
            im = letterbox(im, img_size, stride=stride1, auto=auto)[0]  # padded resize

            # im = im.reshape([1, *im.shape])  # HW to CHW

            im = np.ascontiguousarray(im).astype(np.float32)
            im = im.transpose(2, 0, 1)
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            y = session1.run(output_names, {session1.get_inputs()[0].name: im})
            y2 = session2.run(output_names, {session2.get_inputs()[0].name: im})
            t3 = time.time()

            outputs = nms(torch.tensor(y[0]).cpu().data.numpy(), nclasses=80)
            outputs2 = nms(torch.tensor(y2[0]).cpu().data.numpy(), nclasses=80)

            predictionQueue.put({"m": "prd", "c1": outputs, "c2": outputs2, "frame": element, "names":names1},
                          block=False)

            print("Time inference both models", t3-t_inf)

        except:
            pass
