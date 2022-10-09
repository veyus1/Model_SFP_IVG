# Project

All relevant files used for instance segmentation of droplets in shadowgraphy images from spray-flame synthesis are in this directory.

They have the following usages:

* Model_maskRCNN_droplets.py and Model_maskRCNN_different_backbone.py 🠊 to **train** neural networks w. different backbones respectively, takes data from DropletsMask & DropletsMask_eval directory
* Model_maskRCNN_droplets_visualize_preds.py 🠊 to plot images next to ground truth masks and prediction masks of a specified model, in a loop through dataset; has functionality to note image ids to sort bad ones out, takes data from DropletsMask directory
* Model_maskRCNN_droplets_inference.py and Model_maskRCNN_droplets_inference_cuda.py 🠊 to create predictions from images, no need for gt masks, also returns visible raw images; on cpu or gpu respectively, takes data from Inference directory
* transform_data.py  🠊 transforms pairs of images & masks and transforms both the same and saves transformed versions separately, work in progress but it runs; takes data from DropletsMask_transformed
* Model_ONNX_export.py and Model_ONNX_inference.py 🠊 to convert a set of .pth weights to a ONNX model and run inference on that model (same as above)
* rest is either self explanatory or supplementary modules that were modified one way or another 

current project structure looks like this:

![image](https://user-images.githubusercontent.com/107278273/194773927-955c1e3a-176c-49d0-bb02-60196f07f6c9.png)



 
