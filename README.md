# CSCI 653 - High Performance Computing & Simulations - Final Project

## Integer quantization of NN based Molecular dynamics 
In this project, I would like to explore the possibility of using integer quantization principles towards neural network based molecular dynamics. <br>
Integer quantization has shown to provide up to 16x improvement over traditional FP32 precision computation in neural network for inferences. While higher precision is essential for the training scenario where layer weight updates need to happen in a critical high-precision manner, such precision is generally not essential for inference. Integer quantization, specifically INT8 quantization has been adopted as a solution for various edge-computing paradigms where memory consumption and compute usage is critical. I hope to explore and evaluate such techniques for the field of molecular dynamics. To start with, we can evaluate the behaviour of INT8 quantization and move further towards lower precision (which has been explored in the field of Neural networks, even up to 1 bit quantization).
The goal of this project is thus to enable faster molecular dynamics by making NN based solutions faster to execute. <br>



## Challenges encountered
<li> Most NN based MD solutions work with Graph Neural Networks. Quantization aware training for GNNs is still being researched
<li> However, Quantization aware training has been studied to yield good results even for GNNs, up to INT8 precision - https://openreview.net/forum?id=NSBrFgJAHg
<li> No out-of-the-box quantization solutions for GNNs yet. (Similar alternatives are available for CNNs)
<li> JAX to TF/PyTorch conversion requires deep understanding of the neural network, as anticipated

## Further steps
<li> Attempting to run QAT for tensorflow based molecular dynamics of G-Protein-Coupled receptors (ICNNMD) -  https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c00085
<li> ICNNMD code - https://github.com/Jane-Liu97/ICNNMD
<li> Usage of frameworks like keras make it easier to work with



 ![image](https://user-images.githubusercontent.com/94656693/204927886-dc808f74-6e40-4f6b-9848-bfc2eeec1fa7.png)
  
 ![image](https://user-images.githubusercontent.com/94656693/204927951-4f9fdd06-9eb4-46ae-aeee-ebb80b08ff68.png)

## Proposal
<li> Goal - show faster inference with INT8 inference while maintaining accuracy of results
<li> Methods - NVidia QAT toolkit and run inference on NVidia's TensorRT framework

![image](https://user-images.githubusercontent.com/94656693/204929350-a0f12aeb-0b61-41a5-8dad-c92526d075cc.png)

## Other Curiosities
<li> Use <b> Rust </b> programming language for HPC --> Triple decker program? 
<li> Analyze performance of Rust based implementation
<li> Evaluate <i> "Usability" </i> of Rust for HPC
