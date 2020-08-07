import cv2

'''Video Buffer using cv2'''


def video_processor(input_vid_name):
    vidcap = cv2.VideoCapture(input_vid_name)
    assert vidcap.isOpened()
    width = 0
    height = 0
    frame_count = 0
    img_array = []

    if vidcap.isOpened():

        width = int(vidcap.get(3))
        height = int(vidcap.get(4))
        print('width, height:', width, height)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(width)
    print(height)
    print(frame_count)

    output_writer = cv2.VideoWriter('yolo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_count, (480, 640))  # change output file name if needed
    counter = 2
    while counter <= frame_count:
        success, img = vidcap.read()
        results_1 = cv2.resize(img, (480, 640), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)  # replace with model
        output = cv2.rectangle(results_1, (5, 5), (200, 200), (255, 0, 0), 2)  # random bounding box
        cv2.imwrite("frame%d.jpg" % counter, output)  # save frame as JPEG file
        img_array.append(output)
        counter = counter + 1

    for i in range(len(img_array)):
        output_writer.write(img_array[i])
    cv2.destroyAllWindows()
    output_writer.release()


def main():
    vid_name = "yolo_vid.mp4"  # change input name if needed
    video_processor(vid_name)
    return 0


if __name__ == "__main__":
    main()
