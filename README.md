# How to run the code:
### 1- Result of preprocessing is a file named 'CorrectedDataSet.txt'
### 2- Open and run the file 'Test4Cnn_6.py'



# CBAM
Convolutional bottleneck attention module (CBAM), is a new approach to improve representation power of CNN networks. Attention-based feature refinement is applied with two distinctive modules, channel and spatial, and achieve considerable performance improvement while keeping the overhead small. For the channel attention, it is suggested to use the max-pooled features along with the average-pooled features, leading to produce finer attention. CBAM learns what and where to emphasize or suppress and refines intermediate features effectively and induces the network to focus on target object properly. 

Overal architecture of our proposed method is shown as the following image:


![Alt text](Images/CBAM4.jpg?raw=true "Title")



Feel free to play with different datasets and run CNN & CBAM methods on them.