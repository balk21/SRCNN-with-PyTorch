import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.image as mpimg
import matplotlib.patches as patches
from pathlib import Path

IMAGE_NAME = "ppt3"
CROP_X = 100
CROP_Y = 1000
CROP_SIZE = 100 # 100x100
ZOOM_FACTOR = 10

def create_paper_comparison():
    base_dir = Path("outputs")
    bicubic_path = base_dir / f"{IMAGE_NAME}_bicubic.png"
    srcnn_path = base_dir / f"{IMAGE_NAME}_srcnn.png"
    
    if not bicubic_path.exists() or not srcnn_path.exists():
        print("Couldn't find images in the outputs folder.")
        return

    img_bicubic = mpimg.imread(bicubic_path)
    img_srcnn = mpimg.imread(srcnn_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    def add_zoom_window(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=15)
        ax.axis("off")
        
        rect = patches.Rectangle((CROP_X, CROP_Y), CROP_SIZE, CROP_SIZE, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        axins = zoomed_inset_axes(ax, ZOOM_FACTOR, loc=4) 
        axins.imshow(img)
        
        axins.set_xlim(CROP_X, CROP_X + CROP_SIZE)
        axins.set_ylim(CROP_Y + CROP_SIZE, CROP_Y)
        
        axins.set_xticks([]) 
        axins.set_yticks([])
        
        for spine in axins.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)

    add_zoom_window(ax1, img_bicubic, "Bicubic Upscaling")
    add_zoom_window(ax2, img_srcnn, "SRCNN Reconstruction")

    output_file = base_dir / "final_comparison.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Image created: {output_file}")
    plt.close()

if __name__ == "__main__":
    create_paper_comparison()