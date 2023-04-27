#!/bin/bash

cachePath="$HOME/.cache/torch/hub/checkpoints"
models=("DSFD" "Dlib" "Haar" "MT_CNN" "RetinaFace-ResNet50" "RetinaNetMobileNetV1" "SSD" "YuNet")
testImage=""

for model in "${models[@]}"; do
	echo "$model"
	cd "$model"

	rm -rf model.yml startingPerson.txt *.jpg __pycache__ *.pth *.prototxt *.onnx *.caffemodel *.xml *.dat

	# Haar Cascade
	if [[ "$model" = "Haar" ]]; then
		wget "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
	fi

	# Dlib
	if [[ "$model" = "Dlib" ]]; then
		wget "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
		bzip2 -d ./shape_predictor_68_face_landmarks.dat.bz2
	fi

	# RetinaFace_ResNet50
	if [[ "$model" = "RetinaFace-ResNet50" ]]; then
		wget "https://folk.ntnu.no/haakohu/RetinaFace_ResNet50.pth"
	fi

	# RetinaNetMobileNetV1
	if [[ "$model" = "RetinaNetMobileNetV1" ]]; then
		wget "https://folk.ntnu.no/haakohu/RetinaFace_mobilenet025.pth"
	fi

	if [[ "$model" = "SSD" ]]; then
		wget "https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel?raw=true" -O res10_300x300_ssd_iter_140000_fp16.caffemodel
		wget "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt" -O deploy.prototxt
	fi

	#YuNet
	if [[ "$model" = "YuNet" ]]; then
		wget "https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx?raw=true" -O face_detection_yunet_2022mar.onnx
	fi

	cd ..
done

rm -rf "$cachePath"
mkdir -p "$cachePath"
cp -r ./RetinaFace-ResNet50/RetinaFace_ResNet50.pth "$cachePath"
cp -r ./RetinaNetMobileNetV1/RetinaFace_mobilenet025.pth "$cachePath"

for model in "${models[@]}"; do
	cd "$model"

	file=$(fd . --type f -e py)
	line_number=13
	new_text="testImage = cv2.imread(\"../${testImage}\")"

	awk -v line="$line_number" -v text="$new_text" '{
	  if (NR == line) {
	    print text
	  } else {
	    print $0
	  }
	}' $file >output_file.txt

	mv output_file.txt $file

	python $file

	cd ..
done
