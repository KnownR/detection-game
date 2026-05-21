# classifier_3class.py
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw

class Pose3ClassClassifier:
    def __init__(self, model_path="pose_3class_resnet.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.connections = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),(11,12),(12,14),(14,16),(16,18),(18,20),(20,16),(16,22),(11,23),(12,24),(23,24)]
        self.labels = ['fireball', 'neutral', 'shield']

    def keypoints_to_image(self, landmarks):
        img = Image.new('RGB', (224, 224), (255,255,255))
        draw = ImageDraw.Draw(img)
        pts = [(x*224, y*224) for x,y in landmarks]
        for a,b in self.connections:
            if a < len(pts) and b < len(pts):
                draw.line([pts[a], pts[b]], fill=(0,0,255), width=3)
        for x,y in pts:
            draw.ellipse([(x-6,y-6),(x+6,y+6)], fill=(255,0,0))
        return img

    def predict(self, landmarks):
        img = self.keypoints_to_image(landmarks)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_idx].item()
        return self.labels[pred_idx], conf