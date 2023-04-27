import cv2
import os
import glob
import numpy as np
import dlib


# Detect Face in a given image [EDIT]
def detectFace(testImage):
    grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(testImage)
    return faces, grayImage


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
                        predictor = dlib.shape_predictor(
                            "./shape_predictor_68_face_landmarks.dat"
                        )
                        # Crop region containing face
                        face = faceRect[0]
                        landmarks = predictor(grayImage, face)
                        left = landmarks.part(0).x
                        top = landmarks.part(0).y
                        right = landmarks.part(16).x
                        bottom = landmarks.part(8).y
                        cropFacePart = grayImage[top:bottom, left:right]
                        size = cropFacePart.shape
                        if size[0] > 0 and size[1] > 0:
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
    # faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    ref.update(faces, np.array(labels))
    return ref


def identify(name, trainDir, testImage=None):
    # People Present
    mp = {val: 0 for val in list(name.values())}
    print(mp)

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
        cv2.imwrite("Dlib.jpg", testImage)
        # os.system("brave Dlib.jpg")

    # Draw rectangle around the faces [EDIT]
    def drawRect(testImage, face):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(testImage, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Put Label [EDIT]
    def put_text(testImage, text, x, y):
        cv2.putText(testImage, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

    # Show Test Image Identified People [EDIT]

    if testImage:
        testImage = cv2.imread(testImage)
        ans = ""

        # Faces Detected from test image
        facesDetected, grayImage = detectFace(testImage)
        print(facesDetected)

        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        for face in facesDetected:
            # Give facepart only to the model to predict labels
            landmarks = predictor(grayImage, face)
            left = landmarks.part(0).x
            top = landmarks.part(0).y
            right = landmarks.part(16).x
            bottom = landmarks.part(8).y
            cropFacePart = grayImage[top:bottom, left:right]
            label, confidence = fr.predict(cropFacePart)
            getNum = mp[name[label]]

            if getNum < 1:
                if confidence >= 40:
                    drawRect(testImage, face)
                    personName = name[label]
                    ans += str(personName) + ":"
                    put_text(testImage, personName + f" {confidence}%", left, top - 10)
                    mp[name[label]] += 1

        writeShow(testImage)
        return ans
