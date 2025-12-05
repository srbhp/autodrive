import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, [5, 5], 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # validate slope/intercept
    if not np.isfinite(slope) or not np.isfinite(intercept) or abs(slope) < 1e-6:
        # fall back to a vertical-ish default line in the center quarter
        h, w = image.shape[0], image.shape[1]
        y1 = h
        y2 = int(h * (3 / 5))
        x1 = x2 = int(w * 0.5)
        return np.array([x1, y1, x2, y2])

    h = image.shape[0]
    y1 = h
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # clamp coordinates to image bounds
    w = image.shape[1]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    return np.array([x1, int(y1), x2, y2])


def average_slope_intercept(image, lines, poly_order=2):
    """
    Fit smooth lane curves (polynomial mapping y -> x) from Hough line segments.
    Returns an array-like [left_pts, right_pts] where each is Nx2 (x,y) points
    suitable for cv2.polylines. If fitting fails, returns reasonable defaults.
    """
    h, w = image.shape[0], image.shape[1]
    # default straight lanes (as point arrays)
    def default_lane(x_frac):
        y1 = h
        y2 = int(h * (3 / 5))
        x = int(w * x_frac)
        return np.array([[x, y1], [x, y2]], dtype=np.int32)

    if lines is None or len(lines) == 0:
        return np.array([default_lane(0.25), default_lane(0.75)], dtype=object)

    left_points = []
    right_points = []

    for line in lines:
        try:
            x1, y1, x2, y2 = line.reshape(4)
        except Exception:
            continue
        # skip degenerate segments
        if x1 == x2 and y1 == y2:
            continue
        # classify by slope sign (in image coords y increases downwards)
        if x2 == x1:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2.0
        if slope < 0 and mid_x < w * 0.6:
            left_points.append((mid_x, (y1 + y2) / 2.0))
        elif slope > 0 and mid_x > w * 0.4:
            right_points.append((mid_x, (y1 + y2) / 2.0))
        else:
            # fallback: put into nearest side by x
            if mid_x < w / 2.0:
                left_points.append((mid_x, (y1 + y2) / 2.0))
            else:
                right_points.append((mid_x, (y1 + y2) / 2.0))

    def fit_curve(points):
        # points: list of (x,y) in image coords; we fit x = f(y)
        if len(points) < poly_order + 1:
            return None
        pts = np.array(points)
        xs = pts[:, 0]
        ys = pts[:, 1]
        # fit polynomial mapping y -> x
        try:
            coeffs = np.polyfit(ys, xs, poly_order)
        except Exception:
            return None
        # generate smooth y values and compute x
        y_vals = np.linspace(h, int(h * 0.55), num=100)
        x_vals = np.polyval(coeffs, y_vals)
        pts_out = np.vstack([x_vals, y_vals]).T
        pts_out[:, 0] = np.clip(pts_out[:, 0], 0, w - 1)
        pts_out[:, 1] = np.clip(pts_out[:, 1], 0, h - 1)
        return pts_out.astype(np.int32)

    left_curve = fit_curve(left_points)
    right_curve = fit_curve(right_points)

    if left_curve is None:
        left_curve = default_lane(0.25)
    if right_curve is None:
        right_curve = default_lane(0.75)

    return np.array([left_curve, right_curve], dtype=object)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        # Expect lines to be arrays of points for polylines (Nx2)
        for pts in lines:
            try:
                pts_arr = np.asarray(pts, dtype=np.int32)
            except Exception:
                continue
            if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
                continue
            # reshape for polylines: (num_points,1,2)
            poly = pts_arr.reshape((-1, 1, 2))
            cv2.polylines(line_image, [poly], isClosed=False, color=(0, 255, 255), thickness=6)
    return line_image


def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)  #
    return masked_image


cap = cv2.VideoCapture("./assets/lanes1.mp4")

while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_intrest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
    )
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
