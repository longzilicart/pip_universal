# Prompt-In-Prompt (PIP) Learning for Universal Image Restoration



This repository contains the implementation and demos for paper: "Prompt-In-Prompt Learning for Universal Image Restoration". 


## Abstract
<!-- Image restoration, which aims to retrieve and enhance degraded images, is fundamental across a wide range of applications.
While conventional deep learning approaches have notably improved the image quality across various tasks, they still suffer from （**i**) the high storage cost needed for various task-specific models and (**ii**) the lack of interactivity and flexibility, hindering their wider application. -->
Drawing inspiration from the pronounced success of prompts in both linguistic and visual domains, we propose novel Prompt-In-Prompt learning for universal image restoration, named PIP. First, we present two novel prompts, a degradation-aware prompt to encode high-level degradation knowledge and a basic restoration prompt to provide essential low-level information. Second, we devise a novel prompt-to-prompt interaction module to fuse these two prompts into a universal restoration prompt.
Third, we introduce a selective prompt-to-feature interaction module to modulate the degradation-related feature. By doing so, the resultant PIP works as a plug-and-play module to enhance existing restoration models for universal image restoration.
Extensive experimental results demonstrate the superior performance of PIP on multiple restoration tasks, including image denoising, deraining, dehazing, deblurring, and low-light enhancement.
Remarkably, PIP is interpretable, flexible, robust, and easy-to-use, showing great potential for real-world applications.



<!-- ![flow](figs/flow.png)  -->
<!-- ![pip](figs/pip.png)  -->

<!-- <p float="left">
  <img src=figs/flow.png alt=flow width="50%" />
  <img src=figs/pip.png alt=pip width="49.8%" />
</p> -->


## Advantage
- **Interpretable and flexible**: Prompt-in-prompt learning offers decoupled properties for different degradation types and is flexible for control by both humans and degradation-aware models.
- **Easy-to-use**: PIP is designed as a plug-in-and-play module for existing restoration backbones on the skip-connection. 
- **Robust and efficient**: PIP boosts the backbones, achieving universal restoration at a state-of-the-art level with just a 2.77% increase in parameters.


![decouple](figs/decouple.png)



## Updates

- [x] 23.12.8: initial commit
- [ ] source code coming very soon
- [ ] demos


## Training and testing
The source code is coming very soon.


## Requirements and Data preparation

To set up the environment, please refer to requirements.txt. 


## Citation




## Contact

If you have any question, feel free to contact me at longzilipro@gmail.com






