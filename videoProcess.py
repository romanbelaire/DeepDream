import cv2
import numpy
import dream

def get_frames(path):
    print ('getting frames')
    path = "resources/videos/fish.MP4"

    vid = cv2.VideoCapture(path)
    frames = []

    cap = cv2.VideoCapture(path)

    counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
            cv2.imshow('frame',frame)
            frames.append(frame)
        except:
            counter += 1
            print (counter)
            if(counter == 30):
                break
            continue
        counter = 0
        print('frame')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('got frames')
    return frames

def save_video(frames, path):

    height,width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')#define codec
    out = cv2.VideoWriter(path, fourcc, 20.0, (width, height))
    for frame in frames:
        out.write(frame)#write frame to videos

def play_frames(frames):
    print( 'playing vid')
    for frame in frames:
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    print('played vid')
    cv2.destroyAllWindows()

def cut_frames(frames, interval):
    new_frames = []
    for i, frame in enumerate(frames):
        if i % (interval + 1) == 0:
            new_frames.append(frame)
    return new_frames



#frames = get_frames('asd')
#frames = cut_frames(frames, 4)
#play_frames(frames)
#cv2.imshow('what', dream.dream_image(frames[0], True))
#cv2.waitKey(0)

#frames = dream.dream_video(frames)
frames = dream.dream_video_from_image(cv2.imread('resources/images/tatt.png'), 4)
play_frames(frames)
save_video(frames, 'dream.mp4')
