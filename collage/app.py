import glob

import cv2


def putName(loc, text):
    # Load the image
    img = cv2.imread(loc)

    if text == "YuNet":
        text += "(Chinese Supremacy)"

    elif text == "DSFD":
        text += "(Chinese Supremacy)"

    elif text == "RetinaNetMobileNetV1":
        text += "(US)"

    # Define the font and other properties of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 2
    color = (0, 255, 0)

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Put the text in the top left corner of the image
    x = 10
    y = text_size[1] + 10
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

    # Show the image
    cv2.imwrite(f"./te/{text}U.jpg", img)


for file in glob.glob("./*.jpg"):
    name = file.split("/")[1].split(".")[0]
    putName(file, name)
