# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
#import pyttsx3 as pt
#n = pt.init(driverName='espeak');

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "plant", "sheep",
	"sofa", "train", "monitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	dict = {"L":{"background":0, "aeroplane":0, "bicycle":0, "bird":0, "boat":0,
	"bottle":0, "bus":0, "car":0, "cat":0, "chair":0, "cow":0, "diningtable":0,
	"dog":0, "horse":0, "motorbike":0, "person":0, "plant":0, "sheep":0,
	"sofa":0, "train":0, "monitor":0},"C" : {"background":0, "aeroplane":0, "bicycle":0, "bird":0, "boat":0,
	"bottle":0, "bus":0, "car":0, "cat":0, "chair":0, "cow":0, "diningtable":0,
	"dog":0, "horse":0, "motorbike":0, "person":0, "plant":0, "sheep":0,
	"sofa":0, "train":0, "monitor":0},"R":{"background":0, "aeroplane":0, "bicycle":0, "bird":0, "boat":0,
	"bottle":0, "bus":0, "car":0, "cat":0, "chair":0, "cow":0, "diningtable":0,
	"dog":0, "horse":0, "motorbike":0, "person":0, "plant":0, "sheep":0,
	"sofa":0, "train":0, "monitor":0}}
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	#print(frame.shape)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame,
		0.007843, (300, 300), 127.5)
	# print(blob)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	# print(detections)
	# break
	# loop over the detections
	print("--------------------")
	label_left = []
	label_right = []
	label_center = []
	for i in np.arange(0, detections.shape[2]):
		#count+=1
		
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		#print(confidence , detections.shape[2])
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			#print(count)
			label_cut = label.split(':')
			print(label,"----")
			if((startX+endX)/2<=125):
				print("Left")
				label_t = label.split(':')
				label_left.append(label_t[0])
				if(label_cut[0] in CLASSES):
					dict['L'][label_cut[0]]+=1
			elif((startX+endX)/2 >=175):
				label_t = label.split(':')
				print("Right")
				label_right.append(label_t[0])
				if(label_cut[0] in CLASSES):
					dict['R'][label_cut[0]]+=1
			else:
				label_t = label.split(':')
				print("Center")
				label_center.append(label_t[0])
				if(label_cut[0] in CLASSES):
					dict['C'][label_cut[0]]+=1
			
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			# print("left",label_left)
			# print("right",label_right)
			# print('center',label_center)
			
			print(dict)			

			print("Left label",label_left)
			print("Right label",label_right)
			print("Center label",label_center)
			print("----------------\n",label_left,"\n---------------")
			if(len(label_left)>0):
				label_for_left = label_left[0].split(':')
				print(dict['L'][label_for_left[0]])
			if(len(label_center)>0):
				label_for_center = label_center[0].split(':')
				print(dict['C'][label_for_center[0]])
			if(len(label_right)>0):
				label_for_right = label_right[0].split(':')
				print(dict['R'][label_for_right[0]])
			for j in (label_left):
				if(dict['L'][j]!=0):
					cnt = dict['L'][j]
					if(cnt!=1):
						if(j == 'person'):
							textl = "In Left There are "+str(cnt)+" People"
						elif(j == 'sheep'):
							textl = "In Left There are "+str(cnt)+"Sheep"
						elif(j == 'bus'):
							textl = "In Left There are "+str(cnt)+"buses"
						else:
							textl = "In Left There are "+str(cnt)+" "+str(j)+"s"
					else:
						textl = "In Left There is "+str(cnt)+" "+str(j)
				print(str(textl))
				#n.say(str(textl))
				#n.runAndWait()
			for k in (label_center):
				if(dict['C'][k]!=0):
					cnt = dict['C'][k]
					if(cnt!=1):
						if(k == 'person'):
							textl = "In Left There are "+str(cnt)+" People"
						elif(j == 'sheep'):
							textl = "In Left There are "+str(cnt)+"Sheep"
						elif(j == 'bus'):
							textl = "In Left There are "+str(cnt)+"buses"
						else:
							textc = "In Front there are "+str(cnt)+" "+str(k)+"s"
					else:
						textc = "In Front there is "+str(cnt)+" "+str(k)
				print(str(textc))
				#n.say(str(textc))
				#n.runAndWait()
			for l in (label_right):
				if(dict['R'][l]!=0):
					cnt = dict['R'][l]
					if(cnt!=1):
						if(l == 'person'):
							textl = "In Left There are "+str(cnt)+" People"
						elif(j == 'sheep'):
							textl = "In Left There are "+str(cnt)+"Sheep"
						elif(j == 'bus'):
							textl = "In Left There are "+str(cnt)+"buses"
						else:
							textr = "In Right There are "+str(cnt)+" "+str(l)+"s"
					else:
						textr = "In Right There is "+str(cnt)+" "+str(l)
				print(str(textr))
				#n.say(str(textr))
				#n.runAndWait()
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()