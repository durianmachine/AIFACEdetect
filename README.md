Thank you for downloading the deep learning attendance system. The software and code comes with no warranty and the developer(s) will not be responsible for any damage caused by running or downloading the program. Use at your own risk.

Update notes:
-Added module that detects avaliable computing devices.
  -MPS is not yet support by PyTorch but the code is written out.
  -The program will prioritize CUDA then MPS, then others.
  -MPS is not yet supported by TorchVision but in the case it is detected, it will ask for other computing devices such as CPU.

# AIFACEdetect
Dependencies: 
-PyTorch
-OpenCV
-TorchVision
-Face_Net-Pytorch
-PIL
-MTCNN
-InceptionResnetV1

Requires python pip to install packages.
pip install PyTorch
pip install Opencv
pip install face_net

Requires Python 3.9+

Recommended: Computer with CUDA capability.
Minimum requirement: dual-core CPU with x86 or arm archetecture.

How to use:
After making sure all dependencies are installed and that the Python version is 3.9+:
Clone or download the GitHub repo and place it in a location where it is accessible to the dependencies. As a referene, there is a folder inside the photos folder with Zuckerberg which should detect Zuckerberg. To add additional classes to the face detection, add folders inside the photos folder with the name of the class as the folder name and images in a png format in the corresponding folder. Then, open the main folder in a virtual environment and run the test.py script. From there, it was detect compatible processing devices and auto run.
