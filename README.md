# Keypoints estimation network
A network for hand keypoints estimation in TensorFlow. It can also be used in face landmarks and human pose estimation.

# Reference
The idea is from this paper: [2017 Towards Accurate Multi-person Pose Estimation in the Wild](https://arxiv.org/abs/1701.01779)
This implementation is based on [MobileNet](https://github.com/MG2033/MobileNet).


## Usage
### Main Dependencies
 ```
 Python 3 and above
 tensorflow 1.14.0
 numpy 1.13.1
 tqdm 4.15.0
 easydict 1.7
 matplotlib 2.0.2
 pillow 5.0.0
 ```
### Train and Test
1. Prepare your data, and modify the data_loader.py/DataLoader/load_data() method.
2. Modify the config/test.json to meet your needs.

Note: If you want to test that the model is pretrained and working properly, I've added some test images from different classes in directory 'data/test_images'. All of them are classified correctly.

### Run
```
python3 main.py --config config/test.json

```
The file 'test.json' is just an example of a file. If you run it as is, it will test the model against the images in directory 'data/test_images'. You can create your own configuration file for training/testing.

## Benchmarking
The paper has achieved 569 Mult-Adds. In my implementation, I have achieved approximately 1140 MFLOPS. The paper counts multiplication+addition as one unit. My result verifies the paper as roughly dividing 1140 by 2 is equal to 569 unit.

To calculate the FLOPs in TensorFlow, make sure to set the batch size equal to 1, and execute the following line when the model is loaded into memory.
```
tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
```
I've already implemented this function. It's called ```calculate_flops()``` in `utils.py`. Use it directly if you want.

## Updates
* Inference and training are working properly.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

