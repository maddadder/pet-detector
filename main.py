import PyQt6.QtCore
import PyQt6.QtGui
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QGridLayout, QLabel
from threading import Thread
from collections import deque
from datetime import datetime
import time
import sys
import cv2
import os
import imutils
from dotenv import load_dotenv
load_dotenv('.env')
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from opentimestamps.core.timestamp import Timestamp, OpSet
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import BytesDeserializationContext, BytesSerializationContext, StreamSerializationContext, StreamDeserializationContext, UnsupportedMajorVersion
from opentimestamps.calendar import RemoteCalendar


import io
from queue import Queue, Empty

import hashlib

import logging

from CustomDetachedTimestampFile import CustomDetachedTimestampFile
import tempfile
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
module_handle = "faster_rcnn_openimages_v4_inception_resnet_v2_1"
model = hub.load(module_handle)

# Get the concrete function from the model
detector = model.signatures['default']

frame_skip_interval = 5  # Process every 5th frame

frame_count = 0

HEADER_MAGIC = b'\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94'

def detect_people(frame):

    image = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis]
    # passing the converted image to the model
    detections = detector(image)

    detection_classes = detections["detection_class_entities"]
    detection_scores = detections["detection_scores"]

    detected_people = []

    for i in range(len(detection_scores)):
        #print(detection_classes[i].numpy().decode("ascii") + ":" + str(detection_scores[i].numpy()))
        if detection_scores[i] > 0.3:
            detected_people.append({
                'class': detection_classes[i].numpy().decode("ascii"),
                'score': detection_scores[i].numpy(),
                'bbox': detections['detection_boxes'][i].numpy()
            })

    return detected_people


class CameraWidget(QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into fraame
    """
    
    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, deque_size=1):
        super(CameraWidget, self).__init__(parent)
        
        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)

        # Slight offset is needed since PyQt layouts have a built in padding
        # So add offset to counter the padding 
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        self.camera_stream_link = stream_link
        self.output_dir = "detected"
        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QLabel()
        # Initialize the notary with the appropriate URL
        self.notary_url = "https://a.pool.opentimestamps.org"  # Replace with the correct notary URL

        self.load_network_stream()
        
        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        # Periodically set video frame to display
        self.timer = PyQt6.QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(1)

        print('Started camera: {}'.format(self.camera_stream_link))

    def submit_attestation(self, frame_bytes, timeout):
        """Submit the attestation for a single frame."""
        # Create a SHA256 hash of the frame bytes
        frame_hash = hashlib.sha256(frame_bytes).digest()
        
        # Convert the frame hash to a hexadecimal string
        frame_hash_hex = frame_hash.hex()

        # Write the original frame bytes directly to disk
        image_file_path = os.path.join(self.output_dir, f"{frame_hash_hex}.jpg")  # You can keep the .jpg extension
        with open(image_file_path, 'wb') as f:
            f.write(frame_bytes)
        logging.info(f"Original frame bytes saved to {image_file_path}")

        # Submit the attestation to each calendar URL
        q = Queue()
        self.submit_async(self.notary_url, frame_hash, q, timeout, f"{frame_hash_hex}.ots")

        # Handle response
        try:
            result = q.get(block=True, timeout=timeout)
            if isinstance(result, Timestamp):
                logging.info("Attestation for frame submitted successfully.")
            else:
                logging.warning(f"Submission failed for frame: {result}")
        except Empty:
            logging.warning("Submission timed out for frame.")


    def submit_async(self, calendar_url, attestation_data, queue, timeout, ots_filename):
        """Submit the attestation to the calendar server asynchronously and write the serialized response to a file."""
        logging.debug(f"Submitting attestation to {calendar_url} with data size: {len(attestation_data)} bytes")

        # Log the actual content of attestation_data
        logging.debug(f"Attestation data content: {attestation_data}")

        # Check if attestation_data is empty
        if not attestation_data:
            logging.error("Attestation data is empty.")
            queue.put("Attestation data is empty.")
            return

        try:
            # Create a Timestamp object from attestation data
            timestamp = Timestamp(attestation_data)
            logging.debug(f"Created Timestamp: {timestamp}")

            # Create a RemoteCalendar instance
            calendar = RemoteCalendar(calendar_url)

            # Submit the serialized data to the calendar
            response_timestamp = calendar.submit(timestamp.msg, timeout=timeout)

            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Create the file hash operation (e.g., SHA256)
            file_hash_op = OpSHA256()  # Create an instance of the hash operation

            # Serialize the response timestamp to the output file
            timestamp_file_path = os.path.join(self.output_dir, ots_filename)
            with open(timestamp_file_path, 'xb') as fd:
                fd.write(HEADER_MAGIC)  # Write the header magic bytes
                serialization_ctx = StreamSerializationContext(fd)
                custom_timestamp_file = CustomDetachedTimestampFile(file_hash_op, response_timestamp)  # Pass the hash operation
                custom_timestamp_file.serialize(serialization_ctx)

            logging.info(f"Response written to {timestamp_file_path}")
            queue.put(f"Response written to {timestamp_file_path}")

        except requests.exceptions.Timeout:
            logging.warning("Submission timed out.")
            queue.put("Timeout")
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while submitting attestation: {e}")
            queue.put(str(e))
        except IOError as exp:
            logging.error(f"Failed to create timestamp file: {exp}")
            queue.put(f"Failed to create timestamp file: {exp}")


    def load_network_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.capture.set( cv2.CAP_PROP_FPS, 2 )
                self.online = True
        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def verify_network_stream(self, link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
        
            if self.capture is None:
                self.spin(2)
                continue
            if self.capture.isOpened() and self.online:
                # Read next frame from stream and insert into deque
                status, frame = self.capture.read()
                if status:

                    #Process every frame_skip_interval frame
                    if frame_count % frame_skip_interval != 0:
                        continue

                    
                    # Check if the frame is valid
                    if frame is None or frame.size == 0:
                        logging.warning("Received an empty frame.")
                        continue

                    frame_bytes = None
                    # Convert the frame to bytes (e.g., using JPEG encoding)
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        logging.info("Frame encoded successfully.")
                    except Exception as e:
                        logging.error(f"Error encoding frame: {e}")
                        continue
                        
                    if frame_bytes is not None:
                                                
                        self.submit_attestation(frame_bytes, timeout=10)

                        detected_people = detect_people(frame)

                    for person in detected_people:
                        bbox = person['bbox']
                        ymin, xmin, ymax, xmax = bbox
                        h, w, _ = frame.shape
                        xmin = int(xmin * w)
                        xmax = int(xmax * w)
                        ymin = int(ymin * h)
                        ymax = int(ymax * h)
                        label = person['class'] + ":" + str(person['score'])
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.deque.append(frame)

                    # Automatically skip to the next frame
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()

                else:
                    self.capture.release()
                    self.online = False
            else:
                # Attempt to reconnect
                print('attempting to reconnect', self.camera_stream_link)
                self.load_network_stream()
                self.spin(2)
            self.spin(.001)

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""

        if not self.online:
            self.spin(1)
            return

        if self.deque and self.online:
            # Grab latest frame
            frame = self.deque[-1]

            # Keep frame aspect ratio
            if self.maintain_aspect_ratio:
                self.frame = imutils.resize(frame, width=self.screen_width)
            # Force resize
            else:
                self.frame = cv2.resize(frame, (self.screen_width, self.screen_height))

            # Add timestamp to cameras
            #cv2.rectangle(self.frame, (self.screen_width-190,0), (self.screen_width,50), color=(0,0,0), thickness=-1)
            #cv2.putText(self.frame, datetime.now().strftime('%H:%M:%S'), (self.screen_width-185,37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), lineType=cv2.LINE_AA)

            # Convert to pixmap and set to video frame
            self.img = PyQt6.QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], PyQt6.QtGui.QImage.Format.Format_RGB888).rgbSwapped()
            self.pix = PyQt6.QtGui.QPixmap.fromImage(self.img)
            self.video_frame.setPixmap(self.pix)

    def get_video_frame(self):
        return self.video_frame
    
def exit_application():
    """Exit program event handler"""

    sys.exit(1)

if __name__ == '__main__':

    # Create main application window
    app = QApplication([])
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt())
    #app.setStyle(PyQt6.QtGui.QStyleFactory.create("Cleanlooks"))
    mw = QMainWindow()
    mw.setWindowTitle('Camera GUI')
    #mw.setWindowFlags(PyQt6.QtCore.Qt.FramelessWindowHint)

    cw = QWidget()
    ml = QGridLayout()
    cw.setLayout(ml)
    mw.setCentralWidget(cw)
    mw.showMaximized()
    
    # Dynamically determine screen width/height
    screen_width = QApplication.primaryScreen().geometry().width()
    screen_height = QApplication.primaryScreen().geometry().height()
    
    # Create Camera Widgets 
    username = os.environ.get('USER_NAME')
    password = os.environ.get('PASSWORD')
    password2 = os.environ.get('PASSWORD2')
    
    # Stream links
    camera0 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=1&subtype=1'.format(username, password)
    #camera1 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=2&subtype=1'.format(username, password)
    #camera2 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=3&subtype=1'.format(username, password)
    #camera3 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=4&subtype=1'.format(username, password)
    #camera4 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=5&subtype=1'.format(username, password)
    #camera5 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=6&subtype=1'.format(username, password)
    #camera6 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=7&subtype=1'.format(username, password)
    #camera7 = 'rtsp://{}:{}@192.168.4.137:554/cam/realmonitor?channel=8&subtype=1'.format(username, password)
    #camera8 = 'rtsp://{}:{}@192.168.4.81:554/cam/realmonitor?channel=1&subtype=1'.format(username, password2)
    # Create camera widgets
    print('Creating Camera Widgets...')
    zero = CameraWidget(screen_width//3, screen_height//3, camera0)
    #one = CameraWidget(screen_width//3, screen_height//3, camera1)
    #two = CameraWidget(screen_width//3, screen_height//3, camera2)
    #three = CameraWidget(screen_width//3, screen_height//3, camera3)
    #four = CameraWidget(screen_width//3, screen_height//3, camera4)
    #five = CameraWidget(screen_width//3, screen_height//3, camera5)
    #six = CameraWidget(screen_width//3, screen_height//3, camera6)
    #seven = CameraWidget(screen_width//3, screen_height//3, camera7)
    #eight = CameraWidget(screen_width//3, screen_height//3, camera8)
    
    # Add widgets to layout
    print('Adding widgets to layout...')
    ml.addWidget(zero.get_video_frame(),0,0,1,1)
    #ml.addWidget(one.get_video_frame(),0,1,1,1)
    #ml.addWidget(two.get_video_frame(),0,2,1,1)
    #ml.addWidget(three.get_video_frame(),1,0,1,1)
    #ml.addWidget(four.get_video_frame(),1,1,1,1)
    #ml.addWidget(five.get_video_frame(),1,2,1,1)
    #ml.addWidget(six.get_video_frame(),2,0,1,1)
    #ml.addWidget(seven.get_video_frame(),2,1,1,1)
    #ml.addWidget(eight.get_video_frame(),2,2,1,1)
    print('Verifying camera credentials...')

    mw.show()

    PyQt6.QtGui.QShortcut(PyQt6.QtGui.QKeySequence('Ctrl+Q'), mw, exit_application)

    if(sys.flags.interactive != 1) or not hasattr(PyQt6.QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec()