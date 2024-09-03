
# importing all the libraries
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import  resize, to_pil_image
from torchvision.io import ImageReadMode
from lime import lime_image
from skimage.segmentation import mark_boundaries

from torchcam.methods import SmoothGradCAMpp,GradCAM,XGradCAM
from torch import nn
import torchvision.models as models
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image
import io
from werkzeug.utils import secure_filename

# creating a upload folder
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# the model because when we unload the model in pytorch it requires us to load the architecture of the model as well

# my custom CNN architecrure
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 37 * 37, 160)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(160, 4)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.conv2(x))
        # print(f"x shape:{x.shape}")
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


customCNN = SimpleCNN()

# the fine-tuned efficientNet_b6
efficient_b6 = models.efficientnet_b6();
# the modified fully connected layers
efficient_b6.classifier = nn.Sequential(
    nn.Linear(in_features=2304, out_features=1024, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1024, out_features=4, bias=True),



 )
# the Fine-tuned MobileNet2
mobileNetV2 = models.mobilenet_v2()
# the modified fully connected layers
mobileNetV2.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=512, bias=True),
    nn.ReLU(inplace=True),
   nn.Linear(in_features=512, out_features=4, bias=True),

)
# pinpointing the last layers of each model because we need it for our XAICAM function as we need
# to visualize the results that we get from the last layers of each model.
mobileNet_last_layer=mobileNetV2.features[-1]
customCNN_last_layer = customCNN.conv2
efficient_b6_last_layer = efficient_b6.features[-1]


# loading the three best performing model
# our custom cnn model
state_dict = torch.load('My Custom CNN final with SMOTE.pth',map_location=torch.device('cpu'))
customCNN.load_state_dict(state_dict)
customCNN.eval()



# fine-tuned efficient_b6 model
state_dict1 = torch.load('Efficient_b6_final.pth',map_location=torch.device('cpu'))
efficient_b6.load_state_dict(state_dict1) # getting the weights from the trained model 
efficient_b6.eval()


# fine-tuned mobileNetV2 model
state_dict2= torch.load('MobileNetV2_v2.pth',map_location=torch.device('cpu'))
mobileNetV2.load_state_dict(state_dict2)
mobileNetV2.eval()

# Define the image transformation for custom cnn
transform = transforms.Compose([

    v2.Resize((160, 160)),  # Rescaling
    v2.CenterCrop(size=(150,150)),
    v2.Grayscale(num_output_channels=1),
    transforms.ToTensor(),



])
# image transformation for fine-tuned models
transform2=  transforms.Compose([

    v2.Resize((160, 160)),  # Rescaling
    v2.CenterCrop(size=(150,150)),
    transforms.ToTensor(),
])


#  XAI types
cam_methods = {
    'SmoothGradCAMpp': SmoothGradCAMpp,
    'GradCAM': GradCAM,
    'XGradCAM': XGradCAM,
    'LIME':'LIME'
    
# models
}
models = {
    'efficient_b6':efficient_b6,
    'customCNN':customCNN,
    'mobileNetV2':mobileNetV2
}

# helper functions

# this function converts the plot image  into base64 format and which are then sent to frontend to display
def save_current_plot():
    # Save the current plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0)
    img_bytes.seek(0)
    
    # Encode the image to base64
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

# defining the XAI genral function 
def XAICam(img_path,camtype,ImageMode,model,size:list,target_layer):
    """
    args: img_path is the path where the image is stored
    camtype: the type of CAM technique (XGradCAM,GradCam,SmoothGradCAM) to use
    ImageMode: whether the image is Gray or RGB (1 for Gray and 3 for RGB)
    size: the size of the image to be precessed
    target_layer: the layer which produces the heatmap (usaully points to the final layers)
    """
    if(ImageMode==1): # we need to check this because our custom CNN model only accepts Gray images
        imageMode = ImageReadMode.GRAY
    else:
        imageMode = ImageReadMode.RGB
    img = read_image(img_path,mode=imageMode)
    #preprocessing the image for my model
    input_tensor = resize(img, size).float()
    with camtype(model,target_layer=target_layer) as cam_extractor:
        
        # adding dimention to our input tensor
        out = model(input_tensor.unsqueeze(0))
        # Retrieving the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Visualizing the raw CAM (aka the heatmap)
    plt.imshow(activation_map[0].squeeze(0).numpy(),cmap='inferno')
    plt.axis('off')
    plt.tight_layout()
    img1_base64 = save_current_plot()
    plt.show()
    
    
    
    if ImageMode ==1:
        
        
        rgb_img = to_pil_image(img.repeat(3, 1, 1)) # as our custom cnn model accept only gray image and the overlay_mask only accepts
        # rgb image we need to convert it to rgb to be accepted for overlay_mask
    else:
        
        rgb_img = to_pil_image(img)
    # Resizing the CAM and overlaying it
    result = overlay_mask(rgb_img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Displaying it
    plt.imshow(result,cmap='inferno')
    
    plt.axis('off')
    plt.tight_layout()
    img2_base64 = save_current_plot()
    plt.show()
    
    plt.close('all')
    
    return {
        "base64_raw":img1_base64,
        "base64_cam":img2_base64
    }

#  Lime explanator requires a prediction function
# this function is only needed for explainer.explain_instance
# if we don't have this function the functionality of getting prediction and then display the result for LIME won't work
def getPredictions(imgs,model):
    model_type= models[model]
    
    # print('inside get prediction function',model,imgs)
    if isinstance(imgs, np.ndarray):
        if imgs.ndim == 4:  # batch of images
            imgs = [Image.fromarray((img * 255).astype('uint8'), mode='RGB') for img in imgs]
        elif imgs.ndim == 3:  # single image
            imgs = [Image.fromarray((imgs * 255).astype('uint8'), mode='RGB')]
        else:
            raise ValueError(f"Unexpected number of dimensions: {imgs.ndim}")
    elif isinstance(imgs, Image.Image):
        imgs = [imgs]
#     print(f" image shape:{imgs}")
    processed_imgs = []
    for img in imgs:
        if model!='customCNN':
            # print('not custom')
            img_tensor = transform2(img)
            # print(img_tensor.shape)
        else:
            # print('hello from custom CNN prediction')
            img_tensor = transform(img)   
            # print(img_tensor.shape)
        
        processed_imgs.append(img_tensor)
    
    # Stack the processed images into a batch
    batch = torch.stack(processed_imgs)
#     print(f" batch:{batch} batch shape:{batch.shape}")
    with torch.inference_mode():
        logits = model_type(batch)

        #converting the logits into probabilities using softmax as we have multi-class classification
        prob = torch.softmax(logits,dim=1)    

    return prob.numpy()

def LIMEExplanator (img,model):
    if img.mode != 'RGB': # if the incoming image is not in RGB it will convert it into RGB as explain_intance only accept images
        # that are in RGB
        img = img.convert('RGB')
    img_arr = np.array(img)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_arr, 
                                         lambda x: getPredictions(x, model), # classification function
                                         top_labels=2, # explanation for top class prob
                                         hide_color=0, # hidding part color (0 for black)
                                         num_samples=300)  # number of perturbed samples to explain
    # Let's use mask on image and see the areas that are encouraging the top prediction.
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    
    plt.imshow(img_boundry1,cmap='inferno')
    
    plt.axis('off')
    plt.tight_layout()
    img1_base64 = save_current_plot()
    plt.show()
    img1_base64 = save_current_plot()
    # Let's turn on areas that contributes against the top prediction.
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    
    plt.imshow(img_boundry2,cmap='inferno')
    
    plt.axis('off')
    plt.tight_layout()
    img2_base64 = save_current_plot()
    plt.show()
    plt.close('all')
    
    return {
        "base64_raw":img1_base64,
        "base64_cam":img2_base64
    }
    


 
 





@app.route('/predict', methods=['POST'])
def predict():
    # reading the content sent from frontend
    cam_type = request.form['XAI_technique']
    model_form = request.form['model']
    cam_method = cam_methods[cam_type]
    model_type= models[model_form]

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image'].read() # reading the image
    img = Image.open(io.BytesIO(image)) # as the incoming image is byteIo formate we need to open it using this
    if request.form['model']!='customCNN': # here we check if the model selected is not customCNN
        img = img.convert('RGB') # we convert the image to rgb as the fine-tuned models accept images that are RGB not GRAY
        img_tensor = transform2(img).unsqueeze(0) # transforming the image and adding batch dimension
        print(img_tensor.shape)
    else:
        # print('hello from yes customCNN')
        img_tensor = transform(img).unsqueeze(0)
         
    with torch.inference_mode(): # although the models are already in eval form here we use the inference_mode to save memory, don't calculate the gradient again
        # as the gradients were already calculated in training but now we are in deployment mode
        outputs = model_type(img_tensor) # the outputs of the models are logits not actual values we need to convert them into probabilities
        # pred_prob = torch.softmax(outputs.squeeze(), dim=0)
        _, predicted = torch.max(outputs, 1) # finds the max value index
        probabilities = torch.nn.functional.softmax(outputs, dim=1) # converts the raw logits into probabilities
#         print(f"outputs:{outputs} predicted:{predicted} _:{_} prob:{probabilities} pred_prob:{pred_prob}")
    
    predicted_class = predicted.item() # getting the item from the predicted
    confidence = probabilities[0][predicted_class].item() # getting the confidence value of the predicted class how confident is the
    # model when it predicted that the image belongs to a certain class from 0 to 100%
    # Saving the file to a temporary place
    filename = secure_filename(request.files['image'].filename)
    # the temporary file is stored in Upload_Folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(filepath) # the image is saved to that directory
    
    try:
        # Generating explanation using XAI methods
        if request.form['model']=='customCNN':
            if request.form['XAI_technique'] =='LIME':
                dict_file = LIMEExplanator(img=img,model=model_form)
            else:
                dict_file = XAICam(filepath,cam_method , 1, model_type, (150, 150),customCNN_last_layer)
        elif request.form['model'] =='mobileNetV2':
            if request.form['XAI_technique'] =='LIME':
                dict_file = LIMEExplanator(img=img,model=model_form)
            else:
                dict_file = XAICam(filepath,cam_method , 3, model_type, (150, 150),mobileNet_last_layer)
        else:
            if request.form['XAI_technique'] =='LIME':
                dict_file = LIMEExplanator(img=img,model=model_form)
            else:
                dict_file = XAICam(filepath,cam_method , 3, model_type, (150, 150),efficient_b6_last_layer)
            
    
    finally:
        os.remove(filepath) # as the files are temporary stored in the memory we need to remove it to save memory space

    return jsonify({
        'class': predicted_class,
        'confidence': confidence,
        'image1': dict_file["base64_raw"],
        'image2': dict_file["base64_cam"]
    })
    


    
# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)

