import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit
import cv2

import sys
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils

net_model = 'EfficientNetAutoAttB4'
class prediction:
  def __init__(self, op):
    self.op = op
    # return None
  def check(op):
    # net_model = 'EfficientNetAutoAttB4'

    # Choose a training dataset between
    # - DFDC
    # - FFPP

    train_db = 'DFDC'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224
    frames_per_video = 64

    ## Initialization

    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))
    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)

    face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)

    ## Detect faces

    vid_real_faces = face_extractor.process_video('videos/'+ op)
    # vid_fake_faces = face_extractor.process_video('samples/mqzvfufzoq.mp4')
    # Open the video file
    video = cv2.VideoCapture('videos/'+ op)

    # Find the number of frames in the video file
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Set the frame number to extract (in this case, we want the 10th frame)
    frame_number = 10

    # Set the video to the specified frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = video.read()

    # Check if the frame was extracted successfully
    if success:
        # Save the frame to an image file
        cv2.imwrite('static/Image/frame.jpg', frame)
    else:
        print('Failed to extract frame')

    # Release the video file
    video.release()

           
    im_real_face = vid_real_faces[0]['faces'][0]
   

  

    # For each frame, we consider the face with the highest confidence score found by BlazeFace (= frame['faces'][0])
    faces_real_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_real_faces if len(frame['faces'])] )
    # video.set(cv2.CAP_PROP_POS_FRAMES, faces_real_t )
    # # Read the 10th frame
    # success, frame = video.read()
    # # Save the frame as an image file
    # cv2.imwrite("Image/frame.jpg", frame)
    # faces_fake_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_fake_faces if len(frame['faces'])] )

    with torch.no_grad():
        faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()
        # faces_fake_pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()

    # An average score close to 0 predicts REAL. An average score close to 1 predicts FAKE.

    print('Average score for  video: {:.4f}'.format(expit(faces_real_pred.mean())))
    # print('Average score for FAKE face: {:.4f}'.format(expit(faces_fake_pred.mean())))

    g = expit(faces_real_pred.mean())
    g= g*100
    x = ''
    if g < 50 :
      if g < 10 :
        # print("Video is Fake")
        x = 'Fake'
      if g > 10 :
        # print("Video is Real")
        x = 'Real'
    elif g > 50 :
      # print("Video is Fake")
      x = 'Fake'
    return(x)
