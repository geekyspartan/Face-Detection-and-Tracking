import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    # ROI = Region of interest
    c,r,w,h = window  #c,r,w,h == x,y,w,h #similar to x,y coordinate of a rectangle upper left corner and its width and height
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def camshift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)  #x,y,w,h
    # Our operations on the frame come here
    pt = (frameCounter, c+w/2, r+h/2)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2570, 1 )
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        c,r,w,h = track_window
        pt = cv2.boxPoints(ret)
        pt = np.int0(pt)
        img2 = cv2.polylines(frame,[pt],True, 255,2)
        img2 = cv2.circle(img2, ((c+w/2), (r+h/2)), 5, (0,0,255), -1)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
        

        # write the result to the output file
        x = (frameCounter, c+w/2, r+h/2)
        output.write("%d,%d,%d\n" % x) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

def particle_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    
    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    
    # detect face in first frame
    c,r,w,h = detect_one_face(frame) # c -> x, r -> y

    pt = (0,(c + w/2),(r + h/2))
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    n_particles = 200
    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position

    # Create an array with 200 enteries and each entery is init_pos i.e. [[92 67], [92 67], [92 67]....]
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position

    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
    # Create an array with 200 enteries and each entery is 1/200 i.e. [...]
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)

    stepsize = 8    # pick a value that performs well.
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        # perform the tracking
        # Particle motion model: uniform step (TODO: find a better motion model)
        # For particles, each value of particle, add random number from -step to + step
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        # Clip out-of-bounds particles
        # Clip the particles which goes out of the frame. Set value either 0 of size of frame
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0]))-1).astype(int)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hist_bp = [[0 0 0 ..., 5 7 7]
        # [5 5 5 ..., 6 5 5]
        # [5 5 5 ..., 5 5 5].....]
        hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # particles.T  [[ 80  95  96...][71  59  66  ...]]
        # f  [206 255 122 116...]
        f = particleevaluator(hist_bp, particles.T) # Evaluate particles
        # f.clip(1) -> Replace all 0 with 1 and weights is basically the values returned by particleevaluator -> float value
        weights = np.float32(f.clip(1))
        weights /= np.sum(weights)                  # Normalize w -> divide weight value by total sum
        # Multiply each value of particle.T with corresponding weight value and add both arrays separately (2 arrays in particles.T). sum is the weighted mean which is the new pos
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
        pt = (frameCounter,pos[0], pos[1])
        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y

        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights

        # Display particles
        for sv in particles:
            cv2.circle(frame,(int(sv[0]),int(sv[1])),3,(100,0,255))
        cv2.circle(frame,(int(pos[0]),int(pos[1])),3,(100,0,255))
        cv2.imshow("frame", frame)                         # Plot the image
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",frame)
    
        frameCounter = frameCounter + 1
    
    output.close()

def kalman_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    
    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    #initialize the KF
    kalman = cv2.KalmanFilter(4,2,0)   # 4 - dimensionality of state, 2 - dimensionality of measurement
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                        [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)      # respond faster to change and be less smooth
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    # Write track point for first frame
    pt = (0, c+w/2, r+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        
        #find prediction
        prediction = kalman.predict()

        #obtain measurement
        c,r,w,h = detect_one_face(frame)
        if w != 0 and h != 0:   #measurement_valid
            measurement = np.matrix(np.array([c+w/2, r+h/2], dtype='float64')).transpose()
            posterior = kalman.correct(measurement)
            pt = (frameCounter, int(posterior[0]), int(posterior[1]))
        else:
            # use prediction as the tracking result
            pt = (frameCounter, int(prediction[0]), int(prediction[1]))
        
        img2 = cv2.circle(frame, (pt[1], pt[2]), 5, (0,0,255), -1)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
        
        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
    
    output.close()

def optical_flow_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    
    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    pt = (0, c+w/2, r+h/2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_mask = np.zeros((old_gray.shape[:2]), np.uint8)
    # Take a small patch of the face around the middle area else result will deviate from the actual point.
    x1 = r - 20 +h/2
    x2 = r + 20 +h/2
    y1 = c -15 + w/2
    y2 = c + 15+w/2
    face_mask[x1:x2, y1:y2] = old_gray[x1:x2, y1:y2]

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = face_mask, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        c1,r1,w1,h1 = detect_one_face(frame)
        if c1 !=0 and h1 != 0:
            realPos = [c1 + w1/2,r1 + h1/2]
        else:
            realPos = np.sum(good_new.T, axis = 1)/len(good_new)   # Use optical flow in case face can't be detected.
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv2.circle(frame,(a,b),4,color[i].tolist(),-1)

        img = cv2.add(frame, mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        pt = (frameCounter, realPos[0], realPos[1])
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
    
    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        optical_flow_tracker(video, "output_of.txt")
