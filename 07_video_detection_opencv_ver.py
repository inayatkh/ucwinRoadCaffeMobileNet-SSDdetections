# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


import configparser



def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out[0,0,:,3:7] * np.array([w, h, w, h])

    cls = out[0,0,:,1]
    conf = out[0,0,:,2]

    return (box.astype(np.int32), conf, cls)

def detect(origimg):
	#img = preprocess(origimg)

	(h,w) = origimg.shape[:2]

	blob = cv2.dnn.blobFromImage(cv2.resize(origimg, (300,300)),
		0.007843,(300, 300), 127.5)


	cv2_net.setInput(blob)

	detections = cv2_net.forward()

	#img = img.astype(np.float32)
	#img = img.transpose((2, 0, 1))

	#net.blobs['data'].data[...] = img
	#out = net.forward()  
	box, conf, cls = postprocess(origimg, detections)

	for i in range(len(box)):
		p1 = (box[i][0], box[i][1])
		p2 = (box[i][2], box[i][3])


		if conf[i] > 0.20 :
			c = tuple(map(int, COLORS[int(cls[i])]))
			
			cv2.rectangle(origimg, p1, p2,c,3)

			p3 = (max(p1[0], 15), max(p1[1], 15))
			title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
			cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, c, 2)

	return origimg



if __name__ == '__main__':


	config = configparser.ConfigParser()
	config.read('settings-config.ini')

	train_iter_model = "45000"

	dataset_name = config['DEFAULT']['dataset_name']


	net_file= 'example/MobileNetSSD_deploy.prototxt'  

	caffe_model='deploy/MobileNetSSD_deploy_{}_{}.caffemodel'.format(train_iter_model,dataset_name) 

	test_vid_dir = "/home/inayat/new_retraining_mobilenet/MyDataset/{}/video/test_1.mp4".format(dataset_name)

	#" for frozenElsaDataSet"
	'''
	CLASSES = ('background','bigGirl','smallGirl','dog')

	COLORS = np.array([[255,255,255],
						[255, 0, 255],
						[64, 64, 64],
						[255,153, 51]])

	'''

	# for ucwin objects
	CLASSES = ('background','car','bus','streetlight','person','bike')

	COLORS = np.array([[255,255,255],
						[255, 0, 255],
						[64, 64, 200],
						[255,153, 51],
						[0,255,0],
						[0,0,255]])


	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", required=True,
		help="path to input video file")
	args = vars(ap.parse_args())

	# start the file video stream thread and allow the buffer to
	# start to fill
	print("[INFO] starting video file thread...")
	fvs = FileVideoStream(args["video"]).start()
	time.sleep(1.0)

	# start the FPS timer
	fps = FPS().start()


	##
	FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
	vidWriter =None
	(h,w) = (None, None)
	zeros = None
	out_video_file = '{}_{}.avi'.format(dataset_name,train_iter_model)

	out_fps = 10


	#net = caffe.Net(net_file,caffe_model,caffe.TEST)
	cv2_net = cv2.dnn.readNetFromCaffe(net_file,caffe_model)
	

	while fvs.more():
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale (while still retaining 3
		# channels)
		frame = fvs.read()
		#frame = imutils.resize(frame, width=450)
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#frame = np.dstack([frame, frame, frame])

		# display the size of the queue on the frame

		frame = detect(frame)

		cv2.putText(frame, "Transfer Learning: MobileNet-SSD Detection in Forum-8 UC-Win/Road Virtual Driving Simulator  ",
			(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		cv2.putText(frame, "code available at https://github.com/inayatkh",
			(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)




		# show the frame and update the FPS counter
		cv2.imshow("{} Objects Detections".format(dataset_name), frame)

		if vidWriter is None:
			(h, w) = frame.shape[:2]
			vidWriter = cv2.VideoWriter(out_video_file, FOURCC, out_fps,
				(w,h), True)
			zeros = np.zeros((h,w),dtype="uint8")

		vidWriter.write(frame)

		key = cv2.waitKey(1)

		if key == 27 :
			break

		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()


	vidWriter.release()
		
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	 
	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()
