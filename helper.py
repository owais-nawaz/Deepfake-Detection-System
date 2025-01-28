import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
import face_recognition
from torch import nn
import cv2
import os

# Constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"

train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


# Define the model class
class Model(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
    ):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# Dataset class for video processing
class ValidationDataset(Dataset):
    def __init__(self, video, sequence_length=60, transform=None):
        self.video = video
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1  # Only one video at a time

    def __getitem__(self, idx):
        frames = []
        for i, frame in enumerate(self.frame_extract(self.video)):
            face = self.extract_face(frame)
            if face is not None:
                if self.transform:
                    face = self.transform(face)
                frames.append(face)
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)[: self.count]
        return frames.unsqueeze(0)

    @staticmethod
    def frame_extract(video_path):
        vidObj = cv2.VideoCapture(video_path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

    def extract_face(self, frame):
        """Detect and extract the first face from the frame."""
        faces = face_recognition.face_locations(frame)
        if faces:
            top, right, bottom, left = faces[0]
            return frame[top:bottom, left:right, :]
        return None  # Return None if no face is detected

    def get_face_extracted_frames(self):
        """Extract only faces for displaying purposes."""
        face_frames = []
        for i, frame in enumerate(self.frame_extract(self.video)):
            face = self.extract_face(frame)
            if face is not None:
                face_frames.append(
                    transforms.ToPILImage()(torch.from_numpy(face).permute(2, 0, 1))
                )
            if len(face_frames) == self.count:
                break
        return face_frames


class ValidationDataset_raw(Dataset):
    def __init__(self, video, sequence_length=60, transform=None):
        self.video = video
        self.count = sequence_length

    def __len__(self):
        return 1  # Only one video at a time

    def __getitem__(self, idx):
        frames = []
        for i, frame in enumerate(self.frame_extract(self.video)):
            face = self.extract_face(frame)
            if face is not None:
                frames.append(face)
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)[: self.count]
        return frames.unsqueeze(0)

    @staticmethod
    def frame_extract(video_path):
        vidObj = cv2.VideoCapture(video_path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

    def extract_face(self, frame):
        """Detect and extract the first face from the frame."""
        faces = face_recognition.face_locations(frame)
        if faces:
            top, right, bottom, left = faces[0]
            return frame[top:bottom, left:right, :]
        return None  # Return None if no face is detected

    def get_face_extracted_frames(self):
        """Extract only faces for displaying purposes."""
        face_frames = []
        for i, frame in enumerate(self.frame_extract(self.video)):
            face = self.extract_face(frame)
            if face is not None:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_frames.append(
                    transforms.ToPILImage()(torch.from_numpy(face).permute(2, 0, 1))
                )
            if len(face_frames) == self.count:
                break
        return face_frames


# Model loader
def load_model(model_name):
    model = Model(num_classes=2)  # Adjust the number of classes as necessary
    model_path = os.path.join("./models", model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Softmax function for confidence
def get_prediction_confidence(logits):
    sm = nn.Softmax(dim=1)
    confidence = sm(logits)[0].max().item() * 100
    prediction = "Real" if torch.argmax(logits) == 1 else "Deepfake"
    return prediction, confidence
