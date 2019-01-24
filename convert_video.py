import tensorflow as tf
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip,ImageSequenceClip
import argparse
import os
import shutil

FRONZEN_GRAPH_FILE = "FCN8-optimized.pb"


def process_image(image, logits, keep_prob, image_pl, sess):
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
    
    parser = argparse.ArgumentParser(description='Video Converter')

    parser.add_argument('-f', '--file', type=str,
                        required=True,
                        help="input video")
    parser.add_argument('-r', '--fps', type=int,
                        required=True,
                        help="output frame per sec val")
    parser.add_argument('-s', '--start-frame', type=int,
                        required=False,
                        default=0,
                        help="frame in video to start at")  
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
        
        # Get video
        vid_name = args.file.split('.')[0]
        output = vid_name + "_out.mp4"
        clip = VideoFileClip(args.file).subclip(0,10)
        
        # Create images dir
        if not os.path.isdir(vid_name):
            os.mkdir(vid_name)
        
        # Segment images
        frame_list = list(clip.iter_frames())
        num_frames = len(frame_list)
        i = args.start_frame
        for frame in frame_list[i-1:]:
            print("Converting frame %d of %d" % (i, num_frames))
            seg_image = process_image(frame, logits, keep_prob, image_pl, sess)
            scipy.misc.imsave(os.path.join(vid_name, '{:05}.png'.format(i)), seg_image)
            i +=1            
        
        # Generate clip and cleanup
        resultclip = ImageSequenceClip(vid_name, fps=args.fps)
        resultclip.write_videofile(output, audio=False)
        shutil.rmtree(vid_name)








