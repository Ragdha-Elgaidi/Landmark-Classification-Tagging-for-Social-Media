# Landmark-Classification-Tagging-for-Social-Media
## Project Overview
- Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

- If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgment to classify these landmarks would not be feasible.

## Project Steps
- Create a CNN to Classify Landmarks (from Scratch) - Here, you'll visualize the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. You'll also describe some of your decisions around data processing and how you chose your network architecture. You will then export your best network using Torch Script.

- Create a CNN to Classify Landmarks (using Transfer Learning) - Next, you'll investigate different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network, you'll explain how you arrived at the pre-trained network you chose. You will also export your best transfer learning solution using Torch Script

- Deploy your algorithm in an app - Finally, you will use your best model to create a simple app for others to be able to use your model to find the most likely landmarks depicted in an image. You'll also test out your model yourself and reflect on the strengths and weaknesses of your model.


## Environment and Dependencies
- Download and install Miniconda
- Create a new conda environment with Python 3.7.6:

       conda create --name udacity python=3.7.6 pytorch=1.11.0 torchvision torchaudio cudatoolkit -c pytorch

- Activate the environment:

      conda activate udacity

###### NOTE: You will need to activate your environment again every time you open a new terminal.

- Install the required packages for the project:

      pip install -r requirements.txt

- Test that the GPU is working (execute this only if you have a NVIDIA GPU on your machine, which Nvidia drivers properly installed)

      python -c "import torch;print(torch.cuda.is_available())

- This should return True. If it returns False your GPU cannot be recognized by pytorch. Test with nvidia-smi that your GPU is working. If it is not, check your NVIDIA drivers.

- Install and open jupyter lab:

      pip install jupyterlab 
      jupyter lab

## The Data
- This dataset is called [Landmark Classification](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip) dataset, and contains images of over 50 landmarks.

## CNN from Scratch
- I designed a custom Convolutional Neural Network (CNN) architecture and trained from scratch to classify landmarks. This model was tailored to the specific requirements of the project and underwent rigorous training. It achieved 52% Accuracy.

   - Model Architecture: I decided to use 5 convolutional layers so the model could be sufficiently expressive. I used dropout layers to reduce my model's tendency to overfit the training data. I made my model output a 50-dimensional vector to match with the 50 available landmark classes.
   - Data Preprocessing: My code first resizes the image to 256 and then crops to 224. I picked 224 as the input size because it is the recommended input size for using pytorch's pre-trained models. I did decide to augment the dataset via RandAugment, a typical set of augmentations for natural images. I added this augmentation with the goal of improving my model's robustness, thus improving test accuracy.
    - Training and Validation: I trained for 50 epochs with an adam optimizer and a learning rate scheduler. I saved the weights with the lowest loss
    Accuracy: 52%

## Transfer Learning
- Transfer learning involves leveraging pre-trained CNN models and fine-tuning them for the landmark classification task. This approach capitalizes on the knowledge learned from a large dataset and adapts it to the specific task at hand.
   - Pre-trained Model Selection: I decided to use ResNet50 as the base model. I chose this model because it is a very deep model and it has been trained on a large dataset. I also chose this model because it is a very popular model and I wanted to see how it would perform on this dataset.
   - Training and Validation: Same process as the CNN from scratch.
    Accuracy: 66%
