

This small project can be used to detect 80 classes (coco classes) in a stream or in a video file.
For Decoding I used OPENCV, because there were not requirements for optimiztion,
however in a deployable system I would choose DALI from NVIDIA or PyAv with custom decoding.

I used a process for each major task
    one process for decoding, grabing each frame, decode it and resize
    one process for inference (using 2 different models - a light one and a heavier one)
    one process for visualization (opencv drawing functions can be heavy)

  Decoding process puts the frames in a multiprocessing queue.
  Inference process gets frames from the multiprocessing queue, preprocess the frame, then each model does predictions on that frame sequentially.
  At the end the predictions from both models are put in a multiprocessing queue.
  Visualizer process gets data (frame and predictions from the predictions queue). For each frame we have the bounding boxes, the classes, the confidences.
  In two different windows the output of the models are displayed.

in inference/handler.py CUDA=True or CUDA=False, depending if you want to run in on CPU or GPU
In this solution I used onnx format  and the code can run without GPU, however a better (faster) alternative would be to have a TRITON Server with TENSORRT.
Both models would be on the TRITON Server and we would send requests through grpc.

HOW TO RUN:

pip install -r requirements.txt

python main.py rtsp://86.44.41.160:554/axis-media/media.amp