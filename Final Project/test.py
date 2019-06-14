import cv2 as cv

with open("plate/output.txt") as f:
    lines = f.readlines()

img = None

for l in lines:
    line = l.replace("\n", "")
    text = line.split(" ")

    if len(text) == 1:
        if img is not None:
            cv.imshow("output", img)
            cv.waitKey(0)
        img = cv.imread(f"plate/{text[0]}")
    elif len(text) == 4:
        cv.rectangle(img, (int(text[0]), int(text[1])),
                     (int(text[2]), int(text[3])), (0, 0, 255), 2)

