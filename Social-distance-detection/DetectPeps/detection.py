# import the necessary packages
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0):
	

	#get dimensions dari frame vidio/image dan init list result 
	(H, W) = frame.shape[:2]
	results = []

	#Membuat blop dari frame input dan melakukan pass forward ke YOLO
	#untuk memunculkan bounding person detector
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# inislaisasi list of detected bounding boxes, centroids, and confidences
	boxes = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			
			#Setiap yang terdeteksi akan di beri ClassID dan probabilitasnya
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			#check object detected orang atau tidak
			if classID == personIdx and confidence > MIN_CONF:

				# get center dari object yang terdeteksi (YOLO) 
				# dan memberi bounding ke object(orang) Syang terdeteksi
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				#x dan y untuk size bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update  list of bounding box coordinates,
				# centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# gunakan non-maxima suppression untuk merapikan
	# weak and overlapping bounding boxes	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# Check at least one detection exists
	if len(idxs) > 0:

		# loop over the indexes we are keeping
		for i in idxs.flatten():
			
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# Update results[] yang teridiri dari person probability,
			# coordinates box and centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of result
	return results