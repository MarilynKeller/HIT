import os

from PIL import Image


def make_video(render_folder, output_file, frame_rate, img_string = '%03d.png', white_bg=False):
    
    if white_bg:
        # Load the first image to check the size 
        im_path = os.path.join(render_folder, img_string % 1)
        im = Image.open(im_path)
        im_size = im.size
        im_size_string = f'{im_size[0]}x{im_size[1]}'
    
        cmd = f'/usr/bin/ffmpeg -f lavfi -i color=c=white:s={im_size_string}:r=24 -framerate {frame_rate} -i {render_folder}/{img_string} -y -filter_complex "[0:v][1:v]overlay=shortest=1,format=yuv420p[out]" -map "[out]" {output_file}'
    else:

        # cmd = F'/usr/bin/ffmpeg -framerate {frame_rate} -i {render_folder}/{img_string} -y -vcodec h264 -pix_fmt yuv420p -r 30 -an -b:v 5000k {output_file}'
        # cmd = F'/usr/bin/ffmpeg -i {render_folder}/{img_string} -r 8 -y -c:v libx264 -vf fps={frame_rate} -pix_fmt yuv420p {output_file}'
        cmd = F'/usr/bin/ffmpeg -i {render_folder}/{img_string} -r 8 -y -c:v libx264 -framerate {frame_rate} -pix_fmt yuv420p {output_file}'
        # cmd = F'/usr/bin/ffmpeg -framerate {frame_rate} -i {render_folder}/{img_string} -y -vcodec h264 {output_file}'
        # import ipdb; ipdb.set_trace()
        
    print(cmd)
    # import ipdb; ipdb.set_trace()
    os.system(cmd)

    print(f"Video saved to {output_file}.")

def make_gif(render_folder, output_file, frame_rate, img_string = '%03d.png'):
    
    assert output_file.endswith('.gif'), f'Output file must be a gif, got {output_file}'
    cmd = f'convert -delay 20 -loop 0 -dispose Background {render_folder}/{img_string} {output_file}'
    print(cmd)
    os.system(cmd)

    print(f"Gif saved to {output_file}.")