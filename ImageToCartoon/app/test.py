import cv2
import scipy
from scipy import stats
import numpy as np
from collections import defaultdict


from collections import defaultdict
import numpy as np

def update_c(C, hist):
    # Loop until the bin centers converge
    while True:
        # Initialize an empty dictionary to store pixel indices for each bin
        groups = defaultdict(list)

        # Assign each pixel to the nearest bin center
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)

        # Create a new set of bin centers based on the average pixel intensity in each bin
        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))

        # If the bin centers haven't changed, exit the loop
        if np.sum(new_C - C) == 0:
            break
        C = new_C

    # Return the final set of bin centers and the pixel indices for each bin
    return C, groups


def K_histogram(hist):

    # Set the significance level and minimum number of pixels per bin
    alpha = 0.001
    N = 80

    # Initialize the bin centers to the middle gray value
    C = np.array([128])

    while True:
        # Assign each pixel to the nearest bin center
        C, groups = update_c(C, hist)

        # Create a new set of bin centers
        new_C = set()
        for i, indice in groups.items():
            # Skip bins with fewer than N pixels
            if len(indice) < N:
                new_C.add(C[i])
                continue

            # Test for normality of the pixel intensities in the bin
            z, pval = stats.normaltest(hist[indice])

            # If the pixel intensities are not normal, split the bin into two
            if pval < alpha:
                left = 0 if i == 0 else C[i-1]
                right = len(hist)-1 if i == len(C)-1 else C[i+1]
                delta = right - left

                # If the range of the bin is large enough, split it
                if delta >= 3:
                    c1 = (C[i] + left) / 2
                    c2 = (C[i] + right) / 2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])

        # If the set of bin centers hasn't changed, exit the loop
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))

    return C




def cartoonize_image(img):
    # Define a kernel for erosion
    kernel = np.ones((2, 2), np.uint8)

    # Apply bilateral filter to each channel to smooth the image
    output = np.array(img)
    x, y, c = output.shape
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 150, 150)

    # Apply Canny edge detection to detect edges
    edge = cv2.Canny(output, 100, 200)

    # Convert the image to HSV color space for histogram equalization
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    # Compute histograms for each channel
    hists = []
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)

    # Apply K-means clustering to each histogram to get color centroids
    C = []
    for h in hists:
        C.append(K_histogram(h))

    # Replace each pixel with its nearest color centroid
    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))

    # Convert the image back to RGB color space
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    # Find contours of the edges and draw them on the image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)

    # Apply erosion to smooth out the image
    for i in range(3):
        output[:, :, i] = cv2.erode(output[:, :, i], kernel, iterations=1)

    return output


output = cartoonize_image(cv2.imread("testi.png"))
cv2.imwrite("y.png", output)