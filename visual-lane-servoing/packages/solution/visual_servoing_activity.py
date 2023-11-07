from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    #steer_matrix_left = np.random.rand(*shape)
    
    steer_matrix_left = np.zeros(shape=shape, dtype="float32")
    steer_matrix_left[380:480, 128:256] = -0.75
    steer_matrix_left[440:480, 64:128] = -0.5
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    #steer_matrix_right = np.random.rand(*shape)

    steer_matrix_right = np.zeros(shape=shape, dtype="float32")
    steer_matrix_right[300:480, 256:384] = 0.75
    steer_matrix_right[360:480, 384:475] = 0.5
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # TODO: implement your own solution here
    #mask_left_edge = np.random.rand(h, w)
    #mask_right_edge = np.random.rand(h, w)

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The homography will be made automatically
     
    # Use Gaussian Kernell to blur the image
    sigma = 4
    img_gaussian_filter = cv2.GaussianBlur(img_gray, (0,0), sigma)
    
    # Detect lane markings using sobel edge detection
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)

    # Compute mag of gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute orientation of gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, np.float32), angleInDegrees=True)

    # Find threshold for gradient magnitude, filter out weaker edges
    threshold = 40 #use test image to set
    mask_mag = (Gmag > threshold)

    # Color range for mask
    #white_lower_hsv = np.array([25,0,160])         # CHANGE ME
    #white_upper_hsv = np.array([255, 80, 255])   # CHANGE ME
    #yellow_lower_hsv = np.array([0,90,100])        # CHANGE ME
    #yellow_upper_hsv = np.array([65,255, 255])  # CHANGE ME

    white_lower_hsv = np.array([0, 0, 90]) 
    white_upper_hsv = np.array([120, 50, 255])
    yellow_lower_hsv = np.array([15,50, 50])
    yellow_upper_hsv = np.array([80, 255, 255]) 

    # Create color masks
    mask_white = cv2.inRange(img_hsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(img_hsv, yellow_lower_hsv, yellow_upper_hsv)

    # Left/right edge based masking, may need to comment out depending on usefulness
    width = image.shape[1]
    # image.shape[1]: 440
    mask_left = np.ones(sobelx.shape)   # Initialize left matrix
    mask_left[:,int(np.floor(width/2)):width+1] = 0

    mask_right = np.ones(sobelx.shape)  # Initialize right matrix
    mask_right[:,0:int(np.floor(width/2))] = 0

    # In the left-half image, we are interested in the right-half of the dashed yellow line, which corresponds to negative x- and y-derivatives
    # In the right-half image, we are interested in the left-half of the solid white line, which correspons to a positive x-derivative and a negative y-derivative
    # Generate a mask that identifies pixels based on the sign of their x-derivative
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    shape = image.shape
    mask_left_edge = np.zeros(shape=shape, dtype="float32")     # Initialize left matrix
    mask_right_edge = np.zeros(shape=shape, dtype="float32")    # Initialize right matrix

    mask_left_edge =  mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white


    return mask_left_edge, mask_right_edge
