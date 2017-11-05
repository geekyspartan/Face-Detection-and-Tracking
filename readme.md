# Face Detection and Tracking

Implemented code to detect the face in first frame of the video and then track the face throughout the video using following filters:

* CAMShift
* Particle Filter
* face Detector + Kalman Filter
* Face Detector + Optical Flow Filter

To run code, execute following command

Python command:

> python detection_tracking.py [Filter Number] [Video Path] [Output Path] 

Filter number is following:

* 1 - CAMShift 
* 2 - Particle Filter
* 3 - Kalman Filter
* 4 - Optical Flow Filter

Each filter will save a text file with frame_index and x,y coordinates of the middle of the face/head for each frame in the video.

# Team
[Anurag Arora](https://github.com/geekyspartan)
[Renu Rani](https://github.com/techiepanda)
