import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from configs.cfg import CFG

def get_spatial_transforms():
    # These will be applied to BOTH images identically
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], 
    p=1.0,
    # additional_targets={'image_right': 'image'},
    seed=CFG.SEED 
    )
def get_photometric_transforms():
    # These will be applied INDEPENDENTLY to each half
    return A.Compose([
        A.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.0,p=0.5),
        # A.RandomShadow(
        #     shadow_roi=[0, 0, 1, 1],
        #     num_shadows_limit=[2, 3],
        #     shadow_dimension=5,
        #     shadow_intensity_range=[0.2, 0.5]
        # ),
        # A.RandomSunFlare(
        #     flare_roi=[0.1, 0.1, 0.9, 0.9],
        #     src_radius=20,
        #     src_color=[255, 0, 0],
        #     angle_range=[1, 1],
        #     num_flare_circles_range=[1, 1],
        #     method="physics_based"
        # ),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0, seed=CFG.SEED)

def get_val_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0, seed=CFG.SEED)

train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.0,p=0.5),
        # T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ], p=1.0, seed=CFG.SEED)