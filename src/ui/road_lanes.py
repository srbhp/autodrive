import logging

import cv2
import numpy as np


def grayscale(img):
    """Convert a BGR image to grayscale and validate input.

    OpenCV `VideoCapture` returns BGR images, so use COLOR_BGR2GRAY.
    If `img` is None or empty, return None.
    """
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        logging.error("grayscale: cvtColor failed: %s", e)
        return None
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def polyfit_lanes(input, degree=3, ycrop=0.6):
    """
    TODO:
        - abrupt change in the lanes should be avoided

    """
    # print(input.shape)
    mid = input.shape[1] // 2
    left_only = input[:, 0:mid].nonzero()
    right_only = input[:, mid:].nonzero()
    # add nonzero pixel values and put it in the average value of the pixel indices
    poly_left = np.poly1d(np.polyfit(left_only[0], left_only[1], degree))
    poly_right = np.poly1d(np.polyfit(right_only[0], right_only[1], degree))
    lane_left = poly_left(np.arange(input.shape[0])), np.arange(input.shape[0])
    # shift right lane x values by mid
    lane_right = poly_right(np.arange(input.shape[0])) + mid, np.arange(input.shape[0])

    # use cv2 to show the ploy fit result on the input image
    # fill the area between the lane lines

    out_img = np.dstack((input, input, input)) * 255
    # draw the lane lines , dont use loop

    for y in range(int(input.shape[0] * ycrop), input.shape[0]):
        x_left = int(lane_left[0][y])
        x_right = int(lane_right[0][y])
        out_img[y, x_left:x_right] = [255, 0, 0]
        # draw the lane lines
        out_img[y, x_left - 2 : x_left + 2] = [
            0,
            255,
            0,
        ]  # left lane line in green
        out_img[y, x_right - 2 : x_right + 2] = [
            0,
            0,
            255,
        ]  # right lane line in blue
    return out_img


def weighted_img(img, initial_img, α=0.1, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    # lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges


def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.15, rows]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.95, rows]
    top_right = [cols * 0.55, rows * 0.6]

    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver


# Lane finding Pipeline
def lane_finding_pipeline(image):
    # Validate
    if image is None:
        logging.warning("lane_finding_pipeline: received None image")
        return None

    # Grayscale
    gray_img = grayscale(image)
    if gray_img is None:
        return None
    # Gaussian Smoothing
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    # Canny Edge Detection
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    # Masked Image Within a Polygon
    masked_img = region_of_interest(img=canny_img, vertices=get_vertices(image))
    # Polyfit the lanes
    houghed_lines = polyfit_lanes(masked_img)
    # Draw lines on edges
    output = weighted_img(img=houghed_lines, initial_img=image, α=0.8, β=1.0, γ=0.0)

    return output
