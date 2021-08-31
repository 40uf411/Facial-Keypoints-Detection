import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models import resnet50

class SimpleCNNModel (nn.Module) :
    def __init__(self) :
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 68*2)
    def forward(self, x) :
        output = self.model(x)
        return output

print("[*] Creating the model")
model = SimpleCNNModel().to('cpu')
# load the model checkpoint
checkpoint = torch.load('model.pth', map_location='cpu')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
print("[*] Loaded the model")
model.eval()



# Open the device at the ID 0 
cap = cv2.VideoCapture(0)
#Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")
    
#To set the resolution 
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 220)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 220)

# while(True): 
#     # Capture frame-by-frame

#     ret, frame = cap.read()

#     # Display the resulting frame

#     cv2.imshow('preview',frame)

#     #Waits for a user input to quit the application

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# exit()
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set up the save file path
save_path = f"./vid_keypoint_detection.mp4"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"{save_path}", cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        with torch.no_grad():
            image = frame
            image = cv2.resize(image, (220, 220))
            orig_frame = image.copy()
            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to('cpu')
            outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                        1, (0, 0, 255), -1, cv2.LINE_AA)
        orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
        cv2.imshow('Facial Keypoint Frame', orig_frame)
        out.write(orig_frame)
        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    else: 
        break
    
# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()