import cv2
import multiprocessing
import sys
import setproctitle as setproctitle
import decoding.handler as dec_handler
import inference.handler as inf_handler
import visualizer.handler as vis_handler


if __name__=="__main__":
    setproctitle.setproctitle("Main")
    stream_rtsp = sys.argv[1]

    multiprocessing.set_start_method('spawn')

    framesQueue = multiprocessing.Queue()

    predictionsQueue = multiprocessing.Queue()

    decodingProcess = multiprocessing.Process(target=dec_handler.run, args=(framesQueue, stream_rtsp))
    #
    inferenceProcess = multiprocessing.Process(target=inf_handler.run, args=(framesQueue, predictionsQueue))

    visualizerProcess = multiprocessing.Process(target=vis_handler.run, args=(predictionsQueue,))



    decodingProcess.start()
    inferenceProcess.start()
    visualizerProcess.start()

    decodingProcess.join()
    inferenceProcess.join()


