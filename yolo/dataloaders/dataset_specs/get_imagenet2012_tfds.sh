#!/bin/bash

# email me: vbanna@purdue.edu if there are any issues.
# register your account to download imagenet here http://image-net.org/signup.php?next=download-images

# run instructions:
# > chmod +x get_imagenet2012_tfds.sh
# > sudo ./get_imagenet2012_tfds.sh

ID=dd31405981ef5f776aa17412e1f0c112 #replace this with the ID you recieved when you registered for the dataset, I will sahre the ID but please do not abuse this ID, and do not share the ID

# axel is a library that uses threading to improve download speeds. I highly suggest you download it. This script should handle that
INSTALL_AXEL=1 # if axel is not installed, then install it
AXEL=1 # use axel to load the imagenet zip files over wget. this will reduce download times by a factor of at least 2x
NUM_PARA=4
CHECK_AXEL=$(command -v axel)

#default
PATH_TO_TFDS=~/tensorflow_datasets
DOWNLOADS_DOWNLOAD_DIR=$PATH_TO_TFDS/downloads
MANUAL_DOWNLOAD_DIR=$PATH_TO_TFDS/downloads/manual

TRAIN="http://www.image-net.org/challenges/LSVRC/2012/$ID/ILSVRC2012_img_train.tar"
TEST="http://www.image-net.org/challenges/LSVRC/2012/$ID/ILSVRC2012_img_test_v10102019.tar"
VAL="http://www.image-net.org/challenges/LSVRC/2012/$ID/ILSVRC2012_img_val.tar"
TEST_PATCH="https://drive.google.com/u/0/uc?id=16RYnHpVOW0XKCsn3G3S9GTHUyoV2-4WX&export=download"

# from https://stackoverflow.com/a/13272369
function custom_axel() {
    local file_thingy="$1"
    local url="$2"
    if [ ! -e "$file_thingy" ]; then
        echo "file not found, downloading: $file_thingy"
        axel -avn$NUM_PARA "$url" -o "$file_thingy"
    elif [ -e "${file_thingy}.st" ]; then
        echo "found partial downloaf, resuming: $file_thingy"
        axel -avn$NUM_PARA "$url" -o "$file_thingy"
    else
        echo "already have the file, skipped: $file_thingy"
    fi
}

echo "Links to Imagenet"
echo "train: $TRAIN"
echo "test: $TEST"
echo "validation: $VAL"
echo "(2019) test_patch: $TEST_PATCH"

if  ! [ "${CHECK_AXEL}" ] &> /dev/null && [ $INSTALL_AXEL -eq 1 ]
then
    sudo apt-get install axel
fi

if ! command -v axel &> /dev/null
then
    echo "COMMAND could not be found using wget"
    AXEL=0
fi

mkdir -p $MANUAL_DOWNLOAD_DIR
cd $MANUAL_DOWNLOAD_DIR
pwd

if [ $AXEL -eq 1 ]
then
    echo "downloading using AXEL"
    custom_axel $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_train.tar $TRAIN
else
    wget -c $TRAIN -O $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_train.tar
fi

if [ $AXEL -eq 1 ]
then
    echo "downloading val using AXEL"
    custom_axel $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_val.tar $VAL
else
    wget -c $VAL -O $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_val.tar
fi

if [ $AXEL -eq 1 ]
then
    echo "downloading test using AXEL"
    axel -n $NUM_PARA -a --output=$MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_test.tar $TEST
else
    wget -c $TEST -O $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_test.tar
fi

if [ ! -e $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_test_v10102019.tar ]
then
    wget $VAL -O $MANUAL_DOWNLOAD_DIR/ILSVRC2012_img_test_v10102019.tar
else
    echo "Test patch tar file exists already"
fi

python3 -m tensorflow_datasets.scripts.download_and_prepare --datasets=imagenet2012 --data_dir=$PATH_TO_TFDS --download_dir=$DOWNLOADS_DOWNLOAD_DIR --manual_dir=$MANUAL_DOWNLOAD_DIR
