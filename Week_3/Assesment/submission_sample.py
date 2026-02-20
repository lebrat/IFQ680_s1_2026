import torch
import torch.nn as nn
import torchvision
from PIL import Image
# Import your specific model architecture here

class AircraftClassifier:
    def __init__(self, weights_path):
        # load network architecture and weights here
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)        
        self.model.fc = nn.Linear(self.model.fc.in_features, 20)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        # preprocessing transform
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)), 
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def predict(self, pil_image):
        """
        pil_image: A PIL image
        Output: An integer (0-19) representing the class index
        """
        # preprocess the image and add batch dimension
        image = self.transform(pil_image).unsqueeze(0)         
        with torch.no_grad():
            # Run inference and return the argmax
            outputs = self.model(image)
            predicted = torch.argmax(outputs, 1)
            return predicted.item()


if __name__ == "__main__":
    # Example usage:
    test_image_path = "./FGVCAircraft_Subset20/trainval/class_10/img_0000.jpg"
    pil_image = Image.open(test_image_path).convert("RGB")
    weights_path = "./resnet_frozen_best.pth"           
    model = AircraftClassifier(weights_path=weights_path)
    predicted_class = model.predict(pil_image)
    print(f"Predicted class index: {predicted_class}")