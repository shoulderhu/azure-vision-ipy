import cv2 as cv
import numpy as np
import sys
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte, img_as_bool


def main(imshow=False):

    f = open("output/output.txt", "w+")

    for name in range(1, 21):
        ''' Read Car Image '''
        fname = f"{str(name).zfill(3)}.jpg"
        f.write(f"{fname}\r\n")

        car = cv.imread(f"plate/{fname}")
        if car is None:
            print("Car image is empty")
            sys.exit(0)

        ''' Find Region Of Intrest '''
        car_ori = car[120:490, 320:890]
        car_hsv = cv.cvtColor(car_ori, cv.COLOR_BGR2HSV)
        car_v = cv.split(car_hsv)[2]

        ''' Histogram Equalization'''
        car_eq = cv.equalizeHist(car_v)

        ''' Thresholding '''
        car_th = cv.adaptiveThreshold(car_eq, 255,
                                      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV,
                                      15, -3)

        ''' Cleanup '''
        car_cls = clear_border(img_as_bool(car_th))
        car_cls = img_as_ubyte(remove_small_objects(car_cls, 95))

        if imshow:
            cv.imshow("Thresholding", car_th)
            cv.imshow("Cleanup", car_cls)

        ''' label image regions '''
        car_label = label(car_cls)
        regions = regionprops(car_label)
        regions_idx = []
        bbox = []
        char_total = 0
        height = []

        for idx, region in enumerate(regions):
            if region.area > 350:
                continue

            minr, minc, maxr, maxc = region.bbox
            w, h = maxc - minc, maxr - minr

            # take regions with large enough areas
            if 5 <= w <= 25 and 20 <= h <= 40 and 0.15 <= w / h <= 1:
                height.append(h / 2 + minr)
                regions_idx.append(idx)

        h = np.median(height)
        for idx in regions_idx:
            minr, minc, maxr, maxc = regions[idx].bbox
            if minr <= h <= maxr:
                char_total += 1
                minc += 320
                minr += 120
                maxc += 320
                maxr += 120
                bbox.append([minc, minr, maxc, maxr])
                cv.rectangle(car, (minc, minr), (maxc, maxr), (0, 0, 255), 2)

        bbox.sort(key=lambda x: x[0])
        for box in bbox:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\r\n")

        if imshow:
            cv.imshow(fname, car)
            cv.waitKey(0)
            cv.destroyWindow(fname)

        print(f"{fname}: {char_total} bounding box detected.")
        cv.imwrite(f"plate/{fname}", car)

    f.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main(True)

