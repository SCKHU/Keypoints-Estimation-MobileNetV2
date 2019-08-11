import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# train dataset:
#   image data = 224x224x3 
#   label points = N keypoints
#
# generate_batch_ :
#   image : b x 224 x 224 x 3 (input of network) 
#   label : b x 28 x 28 x 3N (N heantmaps and 2N offset maps)
#   pts : b x N x 2 (key points)


class DataLoader:
    def __init__(self, train_list, test_list, input_size=224, output_size=28, classes=32,
                 batch_size=64, epoch = None, augment = None, normalize=False, shuffle=True):

        self.image_size = input_size
        self.map_size = output_size
        self.classes = classes

        self.train_data_len = 0
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augment = augment
        self.normalize = normalize
        self.epoch = epoch

        self.num_threads = 4
        self.num_prefetch = 5 * self.batch_size

        self.image_list, self.label_list, self.bbox_list = self._read_labeled_image_list(train_list)
        self.test_image_list, self.test_label_list, self.test_bbox_list = self._read_labeled_image_list(test_list)
        
        
    def load_data(self):
        num_channels = 3
        self.train_data_len = len(self.image_list)
        self.test_data_len = len(self.test_image_list)
        return self.image_size, self.image_size, num_channels, self.train_data_len, self.test_data_len
        

    def _read_labeled_image_list(self, data_list):
        f = open(data_list, 'r')
        images = []
        labels = []
        boxes = []
        for line in f:
            tmp = line.strip("\n").split(' ')
            image_path = tmp[0]
            bbox = tmp[1:5]
            bbox = list(map(int, bbox))
            label = tmp[5:(5+self.classes*2)]
            label = list(map(float, label))

            if(bbox[3] > 0 and bbox[2]>0 and bbox[1]>=0 and bbox[0]>=0):
                images.append( image_path)
                labels.append( label)
                boxes.append(bbox)

        return images, labels, boxes


    def generate_batch_(self, type='train'):
        """Reads data, normalizes it, shuffles it, then batches it, returns a
           the next element in dataset op and the dataset initializer op.
           Inputs:
            image_paths: A list of paths to individual images
            label_paths: A list of paths to individual label images
            augment: Boolean, whether to augment data or not
            batch_size: Number of images/labels in each batch returned
            num_threads: Number of parallel calls to make
           Returns:
            next_element: A tensor with shape [2], where next_element[0]
                          is image batch, next_element[1] is the corresponding
                          label batch
            init_op: Data initializer op, needs to be executed in a session
                     for the data queue to be filled up and the next_element op
                     to yield batches"""

        # Convert lists of paths to tensors for tensorflow
        if type == 'train':
            images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
            labels = tf.convert_to_tensor(self.label_list, dtype=tf.float32)
            bbox   = tf.convert_to_tensor(self.bbox_list,  dtype=tf.int32)
            
            data = tf.data.Dataset.from_tensor_slices((images, labels, bbox))
            data = data.shuffle(buffer_size=self.train_data_len)        
            
        else:
            images = tf.convert_to_tensor(self.test_image_list, dtype=tf.string)
            labels = tf.convert_to_tensor(self.test_label_list, dtype=tf.float32)
            bbox   = tf.convert_to_tensor(self.test_bbox_list,  dtype=tf.int32)

            data = tf.data.Dataset.from_tensor_slices((images, labels, bbox))
            data = data.shuffle(buffer_size=self.test_data_len)


        # Parse images and label
        data = data.map(self._parse_data,
                        num_parallel_calls=self.num_threads).prefetch(self.num_prefetch)

        # If augmentation is to be applied
        '''if 'flip_lr' in self.augment:
            #print 'flip_lr'
            data = data.map(self._flip_left_right,
                            num_parallel_calls=self.num_threads).prefetch(self.num_prefetch)'''
        if 'contrast' in self.augment:
            #print 'contrast'
            data = data.map(self._corrupt_brightness,
                            num_parallel_calls=self.num_threads).prefetch(self.num_prefetch)
        if 'saturation' in self.augment:
            #print 'saturation'
            data = data.map(self._corrupt_saturation,
                            num_parallel_calls=self.num_threads).prefetch(self.num_prefetch)
        if 'brightness' in self.augment:
            #print 'brightness'
            data = data.map(self._corrupt_brightness,
                            num_parallel_calls=self.num_threads).prefetch(self.num_prefetch)
        if 'rotate' in self.augment:
            #print 'rotate'
            data = data.map(self._rotate,
                            num_parallel_calls=self.num_threads).prefetch(self.num_prefetch)
     
        # Batch, epoch, shuffle the data
        data = data.batch(self.batch_size, drop_remainder=True)
        data = data.repeat(self.epoch)

        # Create iterator
        iterator = data.make_one_shot_iterator()

        # Next element Op
        next_element = iterator.get_next()
        #init_op = iterator.make_initializer(data)
        return next_element


    def _corrupt_brightness(self, image, label, pts):
        """
        Radnomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_brightness(
            image, 0.4), lambda: tf.identity(image))
        return image, label, pts


    def _corrupt_contrast(self, image, label, pts):
        """
        Randomly applies a random contrast change.
        """
        cond_contrast = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, label, pts


    def _corrupt_saturation(self, image, label, pts):
        """
        Randomly applies a random saturation change.
        """
        cond_saturation = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, label, pts



    def _flip_left_right(self, image, label, pts):

        """Randomly flips image and label left or right in accord."""
        cond_flip = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        
        def fn_true(image, label):
            idx = range(16,-1,-1) + range(26,16,-1) + range(27,31) + range(35,30,-1) + range(45,41,-1) + [47, 46] + range(39, 35,-1) + [41, 40] + range(54,47,-1) + range(59,54,-1) + range(64,59,-1) + [67, 66, 65]
            idx = np.reshape(np.concatenate((np.array(idx)*2,  np.array(idx)*2 + 1),axis=0), [2, -1])
            idx = np.reshape(np.transpose(idx), [-1])
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            label = tf.reshape(tf.concat([
                tf.subtract(1.0, tf.slice(tf.reshape(label,[self.classes, 2]),[0,0],[-1,1])),
                tf.slice(tf.reshape(label,[self.classes, 2]),[0,1],[-1,1])],1),tf.shape(label))
            label = tf.gather(label, idx)
            return image, label
        
        def fn_false(image, label):
            return image, label
        
        image, label = tf.cond(cond_flip, lambda: fn_true(image, label), lambda: fn_false(image, label))
        return image, label, pts


    
    def _rotate(self, image, label, pts):
        cond_rotate = tf.cast(tf.random_uniform(
            [], maxval=5, dtype=tf.int32), tf.bool)
        angle = tf.random_uniform([], minval = -0.5, maxval = 0.5, dtype=tf.float32)
        
        def fn_true(image, label, pts):
            image = tf.contrib.image.rotate(image, angle, interpolation='NEAREST')
            label = tf.contrib.image.rotate(label, angle, interpolation='NEAREST')
            return image, label, pts
        
        def fn_false(image, label, pts):           
            return image, label, pts          
        
        image, label = tf.cond(cond_rotate, lambda: fn_true(image, label), lambda: fn_false(image, label))
        return image, label, pts
    
    #### gaussian heat map ####
    
    def _make_heatmap_gaussian(self, size, pts):
        sigma = 1.5
        heatmap = tf.zeros((size, size, 0))
        for i in range(0, self.classes):
            X1 = tf.linspace(tf.constant(1, tf.float32), tf.cast(size, tf.float32), tf.cast(size, tf.int32 ))
            [X, Y] = tf.meshgrid(X1, X1)
            X -= pts[i, 0] * size
            Y -= pts[i, 1] * size
            D2 = tf.multiply(X, X) + tf.multiply(Y, Y)
            E2 = 2.0 * sigma * sigma
            Exponent = - tf.div(D2, E2)       
            gaussian_map = tf.exp(Exponent)  
            heatmap = tf.concat([heatmap,tf.expand_dims( gaussian_map, 2)], 2)
            
        return heatmap

    #### disk heat map ####
    # size = 224
    def _make_heatmap(self, size, pts):
        R = 2.0
        R2 = R*R
        radius_mask = tf.multiply(tf.ones((size, size), dtype=tf.float32), 
                                  tf.constant(R2, tf.float32))
        #heatmap = tf.zeros((size, size, 0))
        heatmap = self._make_heatmap_gaussian(size, pts)
        offsetmap_x = tf.zeros((size, size, 0))
        offsetmap_y = tf.zeros((size, size, 0))
        X1 = tf.linspace(tf.constant(1, tf.float32), tf.cast(size, tf.float32), tf.cast(size, tf.int32 ))
        [_X, _Y] = tf.meshgrid(X1, X1)
        
        for i in range(0, self.classes):
            X = _X - pts[i, 0] * size
            Y = _Y - pts[i, 1] * size
            X = tf.cast(X, tf.float32)
            Y = tf.cast(Y, tf.float32)
            #D2 = tf.multiply(X, X) + tf.multiply(Y, Y)
            #disk_map = tf.cast(tf.less(D2, radius_mask), tf.float32)
            #heatmap = tf.concat([heatmap, tf.expand_dims( disk_map, 2)], 2)
            
            #offset_map_k = tf.multiply(X, disk_map)
            offset_map_k = tf.divide(tf.clip_by_value(X, -8.0, 8.0),tf.constant(8.0))
            offsetmap_x = tf.concat([offsetmap_x, tf.expand_dims( offset_map_k, 2)], 2)
            
            #offset_map_k = tf.multiply(Y, disk_map)
            offset_map_k = tf.divide(tf.clip_by_value(Y, -8.0, 8.0),tf.constant(8.0))
            offsetmap_y = tf.concat([offsetmap_y, tf.expand_dims( offset_map_k, 2)], 2)
            
        labelmap = tf.concat([heatmap,offsetmap_x,offsetmap_y],2)
        return labelmap


    def _parse_data(self, image_path, pts_orig, bbox):
        """Reads image and label files"""
        pts = pts_orig
        pts = tf.reshape(pts_orig,[self.classes, 2])
        pts = tf.subtract(pts , [bbox[0:2]])
        pts = tf.div(pts , [bbox[2:4]])

        labelmap = self._make_heatmap(self.map_size, pts)

        image_content = tf.read_file(image_path)
        images = tf.image.decode_png(image_content, channels=3)
        images = tf.cast(images , tf.float32)    
        images = tf.image.crop_to_bounding_box(
            image = images,
            offset_height = bbox[1],
            offset_width = bbox[0],
            target_height = bbox[3],
            target_width = bbox[2]
            )
        images = tf.image.resize_images(images, (self.image_size, self.image_size), method=0)

        return images, labelmap, pts  
