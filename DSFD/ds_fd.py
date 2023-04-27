import cv2
import os
import glob
import numpy as np
import face_detection


# Detect Face in a given image [EDIT]
def detectFace(testImage):
    grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=0.5, nms_iou_threshold=0.3
    )
    detections = detector.detect(testImage)
    # print(len(detections))
    # exit(0)
    return detections, grayImage


# Create Labels
def createLables(directory, startingPerson):
    absPath = os.path.abspath(directory)
    persons = glob.glob(f"{absPath}/*")

    faces = []
    labels = []

    for person in persons:
        personID = person.split("/")[-1]

        # Only Process the unprocessed person
        if int(personID) > startingPerson:
            pictures = glob.glob(f"{person}/*")
            print(f"\nProcessing Person{personID}\n")
            for picture in pictures:
                # Check for valid picture
                loadPicture = cv2.imread(picture)

                # Check the shape of the image
                if loadPicture.shape[0] < 40 or loadPicture.shape[1] < 40:
                    print("Error: Invalid image size")
                    continue
                if loadPicture.shape[0] <= 0 or loadPicture.shape[1] <= 0:
                    print("Error: Invalid image size")
                    continue

                # Check the data type of the image
                if loadPicture.dtype != np.uint8:
                    print("Error: Invalid image data type")
                    continue

                # If picture is not valid skip

                if loadPicture is None:
                    print("Cant load this image! Skipping this image...")
                    continue

                # If picture is valid
                else:
                    # Detect the faces from the current training picture
                    faceRect, grayImage = detectFace(loadPicture)

                    # For Debugg Purpose only
                    # showImageRect(loadPicture, faceRect)

                    # Skip the training picture if it has Multiple faces
                    if len(faceRect) > 1:
                        print(
                            f"Image {picture.split('/')[-1]} Contains Multiple faces.Skipping this image..."
                        )
                        continue

                    # Skip if no face detected
                    elif len(faceRect) == 0:
                        print(f"No faces detected in {picture.split('/')[-1]} !")

                    # Process training picture with 1 face [EDIT]
                    elif len(faceRect) == 1:
                        # Crop region containing face
                        pred_bbox = faceRect[0]
                        pred_bbox[2] = pred_bbox[2] - pred_bbox[0]
                        pred_bbox[3] = pred_bbox[3] - pred_bbox[1]
                        pred_bbox = [int(i) for i in pred_bbox[:4]]

                        (x, y, w, h) = pred_bbox
                        cropFacePart = grayImage[y : y + w, x : x + h]
                        size = cropFacePart.shape
                        if cropFacePart.dtype != np.uint8:
                            print("Error: Invalid image data type")
                            continue
                            # Check the shape of the image
                        if cropFacePart.shape[0] < 40 or cropFacePart.shape[1] < 40:
                            print("Error: Invalid image size")
                            continue
                        if cropFacePart.shape[0] <= 0 or cropFacePart.shape[1] <= 0:
                            print("Error: Invalid image size")
                            continue
                        if size[0] <= 0 and size[1] <= 0:
                            print("Invalid Iamge!")
                        if size[0] > 0 and size[1] > 0:
                            print(size[0], size[1])
                            faces.append(np.array(cropFacePart))
                            labels.append(int(personID))
                            print(
                                f"Image {picture.split('/')[-1]} processed successfully!"
                            )

    return faces, labels


# Train classifier
def trainClassifier(faces, labels):
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.train(faces, np.array(labels))
    return faceRecognizer


# Update classifier
def updClassifier(faces, labels, ref):
    ref.update(faces, np.array(labels))
    return ref


def identify(name, trainDir, testImage=None):
    # People Present
    mp = {val: 0 for val in list(name.values())}

    # Handle model check if present or not
    fr = None

    # If startingPerson.txt present that means model is present too
    if os.path.isfile("startingPerson.txt"):
        with open("startingPerson.txt", "r") as f:
            # Get the last processed person id
            startingPerson = int(f.readlines()[-1].lstrip(" ").rstrip(" "))

            # Load Existing model
            existing_model = cv2.face.LBPHFaceRecognizer_create()
            existing_model.read("./model.yml")

            # Train model for any newer data with respect to the last processed person
            faces, labels = createLables(trainDir, startingPerson)

            # If the model do not have current faces update the model and save the model
            if faces or labels:
                ret = updClassifier(faces, labels, existing_model)
                existing_model.save("./model.yml")
                fr = ret

                # Write the last processed person id
                absPath = os.path.abspath(trainDir)
                persons = glob.glob(f"{absPath}/*")
                with open("startingPerson.txt", "w") as f:
                    f.write(f"{len(persons)}")

            # If the model have current faces do not do anything
            else:
                fr = existing_model

    # If startingPerson.txt is not present that means model needs train
    else:
        faces, labels = createLables(trainDir, startingPerson=-1)
        faceRecognizer = trainClassifier(faces, labels)
        faceRecognizer.save("model.yml")
        fr = faceRecognizer
        absPath = os.path.abspath(trainDir)
        persons = glob.glob(f"{absPath}/*")
        with open("startingPerson.txt", "w") as f:
            f.write(f"{len(persons)}")

    # Show Image in browser
    def writeShow(testImage):
        cv2.imwrite("DSFD.jpg", testImage)
        # os.system("brave DSFD.jpg")

    # Draw rectangle around the faces [EDIT]
    def drawRect(testImage, face):
        (x, y, w, h) = face
        cv2.rectangle(testImage, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Put Label [EDIT]
    def put_text(testImage, text, x, y):
        cv2.putText(testImage, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

    if testImage:
        testImage = cv2.imread(testImage)
        ans = ""

        # Faces Detected from test image
        facesDetected, grayImage = detectFace(testImage)
        print(facesDetected)

        # Show Test Image Identified People [EDIT]
        for face in facesDetected:
            # Extract co-ordintes of the face
            pred_bbox = face
            pred_bbox[2] = pred_bbox[2] - pred_bbox[0]
            pred_bbox[3] = pred_bbox[3] - pred_bbox[1]
            pred_bbox = [int(i) for i in pred_bbox[:4]]

            (x, y, w, h) = pred_bbox

            # Give facepart only to the model to predict labels
            facePartOnly = grayImage[y : y + w, x : x + h]
            # Check the shape of the image
            if facePartOnly.shape[0] < 1 or facePartOnly.shape[1] < 1:
                print("Error: Invalid image size")
                continue
            if facePartOnly.shape[0] <= 0 or facePartOnly.shape[1] <= 0:
                print("Error: Invalid image size")
                continue

            # Check the data type of the image
            if facePartOnly.dtype != np.uint8:
                print("Error: Invalid image data type")
                continue
            label, confidence = fr.predict(facePartOnly)
            getNum = mp[name[label]]

            if getNum < 1:
                if confidence >= 40:
                    drawRect(testImage, (x, y, w, h))
                    personName = name[label]
                    ans += str(personName) + ":"
                    put_text(testImage, personName + f" {confidence}%", x, y)
                    mp[name[label]] += 1

        writeShow(testImage)
        return ans
