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

https://projectboard.world/ysc/project/neurologue

How to use:
After making sure all dependencies are installed and that the Python version is 3.9+:
Clone or download the GitHub repo and place it in a location where it is accessible to the dependencies. As a referene, there is a folder inside the photos folder with Zuckerberg which should detect Zuckerberg. To add additional classes to the face detection, add folders inside the photos folder with the name of the class as the folder name and images in a png format in the corresponding folder. Then, open the main folder in a virtual environment and run the test.py script. From there, it was detect compatible processing devices and auto run.

References:
biplob. (2020, December 2). live_face_recognition. GitHub. Retrieved January 14, 2023, from github.com/biplob004/live_face_recognition
Radich, Q., Shemirani, M., & Jenks, A. (2022, June 22). Use pytorch to Train Your Data Analysis Model. Use PyTorch to train your data analysis model | Microsoft Learn. Retrieved January 13, 2023, from learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library [Conference paper]. Advances in Neural Information Processing Systems 32, 8024–8035. papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
Torch Contributors. (2019). Fullyshardeddataparallel. FullyShardedDataParallel - PyTorch 1.11.0 documentation. Retrieved January 15, 2023, from hpytorch.org/docs/1.11/fsdp.html?highlight=fsdp#module-torch.distributed.fsdp
