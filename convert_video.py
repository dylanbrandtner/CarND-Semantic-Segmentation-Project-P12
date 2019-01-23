import tensorflow as tf
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip
import argparse

FRONZEN_GRAPH_FILE = "FCN8-optimized.pb"

#Global tensors
logits = None
keep_prob = None
image_pl = None
sess = None


def process_image(image):

    global logits
    global keep_prob
    global image_pl
    global sess

    desired_shape = (320, 1152)
    original_shape = image.shape
    
    #Resize down to expected size
    image = scipy.misc.imresize(image, desired_shape)
    
    # Run inference
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    # Splice out second column (road), reshape output back to image_shape
    im_softmax = im_softmax[0][:, 1].reshape(image.shape[0], image.shape[1])
    # If road softmax > 0.5, prediction is road
    segmentation = (im_softmax > 0.5).reshape(image.shape[0], image.shape[1], 1)
    # Create mask based on segmentation to apply to original image
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    
    #Resize back up        
    return scipy.misc.imresize(street_im, original_shape)
    
if __name__ == '__main__':

    global logits
    global keep_prob
    global image_pl
    global sess
    
    parser = argparse.ArgumentParser(description='Video Converter')

    parser.add_argument('-f', '--file', type=str,
                        required=True,
                        help="input video")
    args = parser.parse_args()

        
    # delete the current graph    
    tf.reset_default_graph()    

    # Config loader
    config = tf.ConfigProto()
    #jit_level = tf.OptimizerOptions.ON_1
    jit_level = 0
    config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        
        # Load graph
        gd = tf.GraphDef()
        with tf.gfile.Open(FRONZEN_GRAPH_FILE, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        
        #print(len(sess.graph.get_operations()))
        
        # Read in tensors
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        image_pl = sess.graph.get_tensor_by_name('image_input:0')
        logits = sess.graph.get_tensor_by_name('Reshape:0')
        
        # Convert video
        output = args.file.split('.')[0] + "_out.mp4"
        clip1 = VideoFileClip(args.file)        
        proc_clip = clip1.fl_image(process_image) 
        proc_clip.write_videofile(output, audio=False)








