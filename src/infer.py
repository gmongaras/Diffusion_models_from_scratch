import torch
from .models.diff_model import diff_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import click






@click.command()

# Required
@click.option("--loadDir", "loadDir", type=str, help="Location of the models to load in.", required=True)
@click.option("--loadFile", "loadFile", type=str, help="Name of the .pkl model file to load in. Ex: model_358e_450000s.pkl", required=True)
@click.option("--loadDefFile", "loadDefFile", type=str, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl", required=True)

# Generation parameters
@click.option("--step_size", "step_size", type=int, default=10, help="Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.", required=False)
@click.option("--DDIM_scale", "DDIM_scale", type=int, default=0, help="Must be >= 0. When this value is 0, DDIM is used. When this value is 1, DDPM is used. A low scalar performs better with a high step size and a high scalar performs better with a low step size.", required=False)
@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
@click.option("--guidance", "w", type=int, default=4, help="Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.", required=False)
@click.option("--class_label", "class_label", type=int, default=0, help="0-indexed class value. Use -1 for a random class and any other class value >= 0 for the other classes. FOr imagenet, the class value range from 0 to 999 and can be found in data/class_information.txt", required=False)
@click.option("--corrected", "corrected", type=bool, default=False, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)

# Output parameters
@click.option("--out_imgname", "out_imgname", type=str, default="fig.png", help="Name of the file to save the output image to.", required=False)
@click.option("--out_gifname", "out_gifname", type=str, default="diffusion.gif", help="Name of the file to save the output image to.", required=False)
@click.option("--gif_fps", "gif_fps", type=int, default=10, help="FPS for the output gif.", required=False)

def infer(
    loadDir: str,
    loadFile: str,
    loadDefFile: str,

    step_size: int,
    DDIM_scale: int,
    device: str,
    w: int,
    class_label: int,
    corrected: bool,

    out_imgname: str,
    out_gifname: str,
    gif_fps: int
    ):

    
    
    
    
    
    ### Model Creation

    # Create a dummy model
    model = diff_model(3, 3, 1, 1, ["res", "res"], 100000, "cosine", 100, device, 100, 1000, 16, 0.0, step_size, DDIM_scale)
    
    # Load in the model weights
    model.loadModel(loadDir, loadFile, loadDefFile)
    
    # Sample the model
    noise, imgs = model.sample_imgs(1, class_label, w, True, True, True, corrected)
            
    # Convert the sample image to 0->255
    # and show it
    plt.close('all')
    plt.axis('off')
    noise = torch.clamp(noise.cpu().detach().int(), 0, 255)
    for img in noise:
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig(out_imgname, bbox_inches='tight', pad_inches=0, )
        plt.show()

    # Image evolution gif
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_axis_off()
    for i in range(0, len(imgs)):
        title = plt.text(imgs[i].shape[0]//2, -5, f"t = {i}", ha='center')
        imgs[i] = [plt.imshow(imgs[i], animated=True), title]
    animate = animation.ArtistAnimation(fig, imgs, interval=1, blit=True, repeat_delay=1000)
    animate.save(out_gifname, writer=animation.PillowWriter(fps=gif_fps))
    
    
    
    
    
if __name__ == '__main__':
    infer()