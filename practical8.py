import cv2
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# load model
model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     # Downloads pre-trained features
    in_channels=3,                  
    classes=1                       
)
model.eval()

# read image
img = cv2.imread('drama.png')
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to tensor
img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# prediction
with torch.no_grad():
    mask = model(img_tensor)
    mask = torch.sigmoid(mask)        # apply sigmoid here if not in model
    mask_np = mask[0, 0].numpy()

# visualization
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(mask_np, cmap="gray")

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()