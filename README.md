# FaceGenerationFromText
Automatically synthesizing images with the use of text is quite an interesting task and a lot of research is being done on the same as the current AI systems seem quite far from 
this goal. This repo contains an experimental implementation of generating facial images using unstructured textual descriptions, making use of tensorflow. Two different approaches
based on GAN-CLS algorithm and StackGAN have been explored. Our GAN models have been trained on the CelebA dataset. 
The folder structure of the repo is explained below:
- **CaptionGeneration** : Contains code for capturing semantic information about a face, given by its attributes, into meaningful captions.
- **GAN-CLS** : Contains code for GAN-CLS based model
- **StackGAN** : Contains code for StackGAN based model

### Installation Requirements ###
- python
- tensorflow
- h5py
- Theano
- scikit-learn
- NLTK

### Usage ###

- **Caption Generation**

    - Move into the CaptionGeneration folder using the cd command.
    - Run "python caption_generator.py"
    
- **Skip-thought Encoding Generation**

    - Run "python generate_thought_vectors.py --caption-file = 'CaptionGeneration/captions.txt' " 

- **Training GAN-CLS based Model**

    - Move in the directory, using the cd command, into the GAN-CLS folder.
    - Run "python train.py"
   
- **Training StackGAN based Model**

    - For training Stage-1, move into stage1 folder in StackGAN using cd command and run "python stage1_train.py"
    - For training Stage-2, move into stage2 folder in StackGAN using cd command and run "python stage2_train.py"
    
- **Generating Images from Captions**

    - Write the captions in a text file, and save them as demo_captions.txt. Generate the skip thought vectors for these captions.
    - Generate images corresponding to these captions using the thought vectors and run "python generate_images.py --model-path = <path to trained model>"
        
    
 
