#Modules for Pi
import os
import time
from gpiozero import Button
import RPi.GPIO as GPIO

#Modules for Object Detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3 as pt

#Modules for Text Recognition
import logging
import shutil
import subprocess
import sys
import tempfile
from constants import VALID_IMAGE_EXTENSIONS, WINDOWS_CHECK_COMMAND, DEFAULT_CHECK_COMMAND, TESSERACT_DATA_PATH_VAR

GPIO.setmode(GPIO.BCM)

GPIO.setup(5, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.setup(6, GPIO.IN, pull_up_down=GPIO.PUD_UP)
  
n = pt.init(driverName='espeak');

def observeMode():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    prototext = "MobileNetSSD_deploy.prototxt.txt"
    model = "MobileNetSSD_deploy.caffemodel"

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "plant", "sheep",
        "sofa", "train", "monitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototext, model)
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
                    n.say(str(textl))
                    n.runAndWait()
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
                    n.say(str(textc))
                    n.runAndWait()
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
                    n.say(str(textr))
                    n.runAndWait()
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

def create_directory(path):
        """
        Create directory at given path if directory does not exist
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)

def check_path(path):
        """
        Check if file path exists or not
        :param path:
        :return: boolean
        """
        return bool(os.path.exists(path))

def get_command():
        """
        Check OS and return command to identify if tesseract is installed or not
        :return:
        """
        if sys.platform.startswith('win'):
            return WINDOWS_CHECK_COMMAND
        return DEFAULT_CHECK_COMMAND

def run_tesseract(filename, output_path, image_file_name):
    # Run tesseract
    filename_without_extension = os.path.splitext(filename)[0]
    # If no output path is provided
    if not output_path:
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, filename_without_extension)
        subprocess.run(['tesseract', image_file_name, temp_file],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        with open('{}.txt'.format(temp_file), 'r', encoding="utf8") as f:
            text = f.read()
        shutil.rmtree(temp_dir)
        return text
    text_file_path = os.path.join(output_path, filename_without_extension)
    subprocess.run(['tesseract', image_file_name, text_file_path],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    return

def check_pre_requisites_tesseract():
    """
    Check if the pre-requisites required for running the tesseract application are satisfied or not
    :param : NA
    :return: boolean
    """
    check_command = get_command()
    logging.debug("Running `{}` to check if tesseract is installed or not.".format(check_command))

    result = subprocess.run([check_command, 'tesseract'], stdout=subprocess.PIPE)
    if not result.stdout:
        logging.error("tesseract-ocr missing, install `tesseract` to resolve. Refer to README for more instructions.")
        return False
    logging.debug("Tesseract correctly installed!\n")

    if sys.platform.startswith('win'):
        environment_variables = os.environ
        logging.debug(
            "Checking if the Tesseract Data path is set correctly or not.\n")
        if TESSERACT_DATA_PATH_VAR in environment_variables:
            if environment_variables[TESSERACT_DATA_PATH_VAR]:
                path = environment_variables[TESSERACT_DATA_PATH_VAR]
                logging.debug("Checking if the path configured for Tesseract Data Environment variable `{}` \
                as `{}` is valid or not.".format(TESSERACT_DATA_PATH_VAR, path))
                if os.path.isdir(path) and os.access(path, os.R_OK):
                    logging.debug("All set to go!")
                    return True
                else:
                    logging.error(
                        "Configured path for Tesseract data is not accessible!")
                    return False
            else:
                logging.error("Tesseract Data path Environment variable '{}' configured to an empty string!\
                ".format(TESSERACT_DATA_PATH_VAR))
                return False
        else:
            logging.error("Tesseract Data path Environment variable '{}' needs to be configured to point to\
            the tessdata!".format(TESSERACT_DATA_PATH_VAR))
            return False
    else:
        return True

def main(input_path, output_path):
    # Check if tesseract is installed or not
    if not check_pre_requisites_tesseract():
        return

    # Check if a valid input directory is given or not
    if not check_path(input_path):
        logging.error("Nothing found at `{}`".format(input_path))
        return

    # Create output directory
    if output_path:
        create_directory(output_path)
        logging.debug("Creating Output Path {}".format(output_path))

    # Check if input_path is directory or file
    if os.path.isdir(input_path):
        logging.debug("The Input Path is a directory.")
        # Check if input directory is empty or not
        total_file_count = len(os.listdir(input_path))
        if total_file_count == 0:
            logging.error("No files found at your input location")
            return

        # Iterate over all images in the input directory
        # and get text from each image
        other_files = 0
        successful_files = 0
        logging.info("Found total {} file(s)\n".format(total_file_count))
        for ctr, filename in enumerate(os.listdir(input_path)):
            logging.debug("Parsing {}".format(filename))
            extension = os.path.splitext(filename)[1]

            if extension.lower() not in VALID_IMAGE_EXTENSIONS:
                other_files += 1
                continue

            image_file_name = os.path.join(input_path, filename)
            print(run_tesseract(filename, output_path, image_file_name))
            successful_files += 1

        logging.info("Parsing Completed!\n")
        if successful_files == 0:
            logging.error("No valid image file found.")
            logging.error("Supported formats: [{}]".format(
                ", ".join(VALID_IMAGE_EXTENSIONS)))
        else:
            logging.info(
                "Successfully parsed images: {}".format(successful_files))
            logging.info(
                "Files with unsupported file extensions: {}".format(other_files))

    else:
        filename = os.path.basename(input_path)
        logging.debug("The Input Path is a file {}".format(filename))
        text = run_tesseract(filename, output_path, input_path) 
        print(text)
        n.say(text);
        n.runAndWait()

def readMode():
    n = pt.init(driverName='espeak');
    path = "/home/nisarg/image2text-master/realtime.jpg"
    #os.system("cheese")
    os.system("fswebcam -r 640x480 --jpeg 85 -D 0 "+path)
    os.system("eog /home/nisarg/image2text-master/realtime.jpg")
    
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    #required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    #required.add_argument('-i', '--input', help="Single image file path or images directory path", required=True)
    optional.add_argument('-o', '--output', help="(Optional) Output directory for converted text")
    optional.add_argument('-d', '--debug', action='store_true', help="Enable verbose DEBUG logging")

    args = parser.parse_args()
    #input_path = os.path.abspath(args.input)
    input_path = path
    #print("Args : ",args)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = None

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.debug("Input Path is {}".format(input_path))

    # Check Python version
    if sys.version_info[0] < 3:
        logging.error("You are using Python {0}.{1}. Please use Python>=3".format(
            sys.version_info[0], sys.version_info[1]))
        exit()

    main(input_path, output_path)

while(True):
    if GPIO.input(5)==0:
        observeMode()
        #time.sleep(3)
    elif GPIO.input(6)==0:
        readMode()