# CSCI 653 - High Performance Computing & Simulations - Final Project

## Integer quantization of NN based Molecular dynamics 
In this project, I would like to explore the possibility of using integer quantization principles towards neural network based molecular dynamics. <br>
Integer quantization has shown to provide up to 16x improvement over traditional FP32 precision computation in neural network for inferences. While higher precision is essential for the training scenario where layer weight updates need to happen in a critical high-precision manner, such precision is generally not essential for inference. Integer quantization, specifically INT8 quantization has been adopted as a solution for various edge-computing paradigms where memory consumption and compute usage is critical. I hope to explore and evaluate such techniques for the field of molecular dynamics. To start with, we can evaluate the behaviour of INT8 quantization and move further towards lower precision (which has been explored in the field of Neural networks, even up to 1 bit quantization).
The goal of this project is thus to enable faster molecular dynamics by making NN based solutions faster to execute. <br>


## Current Status
I was able to perform integer quantization for the TensorFlow model of Molecular Dynamics of G-Protein Coupled Receptors. This model was obtained from the works of (ICNNMD mentioned below).<br>
Inference with the integer quantized model ran x faster than the FP32 version with same accuracy. This shows potential for integer quantization techniques to be adopted for NN based MD. <br><br>
Further details on experiments are in the sections after proposal. 

## Challenges encountered
<li> Most NN based MD solutions work with Graph Neural Networks. Quantization aware training for GNNs is still being researched
<li> However, Quantization aware training has been studied to yield good results even for GNNs, up to INT8 precision - https://openreview.net/forum?id=NSBrFgJAHg
<li> No out-of-the-box quantization solutions for GNNs yet. (Similar alternatives are available for CNNs)
<li> JAX to TF/PyTorch conversion requires deep understanding of the neural network, as anticipated

## Project Strategy
<li> Attempting to run QAT for tensorflow based molecular dynamics of G-Protein-Coupled receptors (ICNNMD) -  https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c00085
<li> ICNNMD code - https://github.com/Jane-Liu97/ICNNMD
<li> Usage of frameworks like keras make it easier to work with



 ![image](https://user-images.githubusercontent.com/94656693/204927886-dc808f74-6e40-4f6b-9848-bfc2eeec1fa7.png)
  
 ![image](https://user-images.githubusercontent.com/94656693/204927951-4f9fdd06-9eb4-46ae-aeee-ebb80b08ff68.png)

## Proposal
<li> Goal - show faster inference with INT8 inference while maintaining accuracy of results
<li> Methods - NVidia QAT toolkit and run inference on NVidia's TensorRT framework

![image](https://user-images.githubusercontent.com/94656693/204929350-a0f12aeb-0b61-41a5-8dad-c92526d075cc.png)


## Workflow Design
By adopting the ICNNMD codebase, I was able to come up with the following high level workflow for the quantization task. 

![image](https://user-images.githubusercontent.com/94656693/207797844-fc08591d-3bc6-4925-a3bc-6cd0931c3ccb.png)

## Experiment Setup
 
### Dataset
 Protein Data Bank and NC Trajectory files for Molecular Dynamics simulation of G-Protein Coupled receptor - "Chimera protein of cc chemokine receptor type 2 isoform b and t4-lysozyme (apoform) (apoform)" <br>
 Source - https://submission.gpcrmd.org/dynadb/dynamics/id/734/

### Environment
 | Resource | Type |
| --- | --- |
| GPU | NVidia Volta V100 x 4 |
| CPU | Intel Xeon E5-2680 |
 
 
  <li> Tensorflow 2.x w/ Keras
  <li> TensorRT Engine for Inference
  <li> NVidia Tensorflow Quantization Toolkit for quantization aware training
  
 
 ## Experiment Results
   I was able to train the model to initially obtain the FP32 model. Then, further fine tune training with quantization/dequantization nodes was performed with NVidia Toolkit to obtain the integer quantized mode. <br>This model was then fed to trt engine creation logic to obtain Tensor RT engine for inference.
   
 ### Accuracy
   | Model | Train Accuracy | Test Accuracy |
| --- | --- | --- |
| FP16 | 1.0 | 0.93 |
| INT8 | 1.0 | 0.93 |
   
 ### Timings
 | Model | H2D_Timings | GPU_Compute | D2HTimings |
 | --- | --- | --- | --- |
 | FP16	| 0.065 ms |	0.166 ms |	0.005 ms |
 | INT8	| 0.063	ms | 0.042	ms | 0.013 ms |
   
 
 ![image](https://user-images.githubusercontent.com/94656693/207800278-3e10d09b-d952-44be-bc13-056dc3ce0aaa.png)

 ### Network comparison
   Following comparison was generated with NVidia's Tensor Engine Explorer utility to understand the specific functions used by TensorRT in every layer of the NN and the dimensions of the inputs/outputs for every layer. 
   <br> We can see here that the INT8 model quantizes (scales) inputs initially and then works only with INT8 precision while FP16 model sticks with FP16 precision
   
   


| FP16 | INT8 |
:-------------------------:|:-------------------------:
![FP16_Network](https://user-images.githubusercontent.com/94656693/207801244-3e9ea794-6b11-4058-ba7f-6e3af913501a.jpg)  |  ![INT8_Network](https://user-images.githubusercontent.com/94656693/207801315-328f178b-8866-4d46-adf7-5e0df21b92e9.jpg)
   
 ## Conclusion
   The above experiment results show a 75% reduction in inference timings for molecular dynamics simulation for one time-step while retaining accuracy. <br>
   While this particular molecule and neural network might not be the best performant, this serves as an indication that integer quantization can help in molecular dynamics scenario as well.
   Further experiment result screenshots are available in the repo.

   
   
   
