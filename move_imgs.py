import shutil
from pathlib import Path
from tqdm import tqdm  # Run `pip install tqdm` if you don't have this

def copy_images(image_paths, destination_folder):
    """
    Copies a list of images to a destination folder.
    
    Args:
        image_paths (list): List of file paths (strings or Path objects).
        destination_folder (str): Path to the target directory.
    """
    # 1. Create the destination folder if it doesn't exist
    dest_path = Path(destination_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {len(image_paths)} images to '{dest_path.resolve()}'...")

    success_count = 0
    errors = []

    # 2. Loop through paths and copy
    for img_path in tqdm(image_paths, unit="img"):
        try:
            # Convert string path to Path object
            src = Path('csiro-biomass/'+img_path)
            
            # Check if source exists before trying to copy
            if not src.exists():
                errors.append(f"Source not found: {src}")
                continue
                
            # Define destination file path (keeps the original filename)
            # If you want to handle duplicate filenames, you'd add logic here
            dst = dest_path / src.name
            
            # Perform the copy
            # shutil.copy2 preserves metadata (timestamps, etc.)
            shutil.copy2(src, dst)
            success_count += 1
            
        except Exception as e:
            errors.append(f"Error copying {src}: {e}")

    # 3. Summary Report
    print(f"\n--- Summary ---")
    print(f"Successfully copied: {success_count}")
    print(f"Failed: {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f" - {err}")

# --- Example Usage ---
if __name__ == "__main__":
    # Example list of paths (can be absolute or relative)
    my_images = ['train/ID1028611175.jpg', 'train/ID1035947949.jpg', 'train/ID1049634115.jpg', 'train/ID1052620238.jpg', 'train/ID1108283583.jpg', 'train/ID1119739385.jpg', 'train/ID1127246618.jpg', 'train/ID1136169672.jpg', 'train/ID1148528732.jpg', 'train/ID1183807388.jpg', 'train/ID1193692654.jpg', 'train/ID1215977190.jpg', 'train/ID12390962.jpg', 'train/ID1277756619.jpg', 'train/ID1291116815.jpg', 'train/ID1450399782.jpg', 'train/ID1474775613.jpg', 'train/ID1512751450.jpg', 'train/ID1515990019.jpg', 'train/ID1525817840.jpg', 'train/ID1547945326.jpg', 'train/ID1618597318.jpg', 'train/ID1781353117.jpg', 'train/ID1789853061.jpg', 'train/ID1839139621.jpg', 'train/ID1853508321.jpg', 'train/ID1859792585.jpg', 'train/ID1868719645.jpg', 'train/ID1880764911.jpg', 'train/ID1920959057.jpg', 'train/ID196516535.jpg', 'train/ID1988033238.jpg', 'train/ID1993907137.jpg', 'train/ID2003438517.jpg', 'train/ID2099742797.jpg', 'train/ID210865340.jpg', 'train/ID2145635095.jpg', 'train/ID290369222.jpg', 'train/ID315357834.jpg', 'train/ID332742639.jpg', 'train/ID344618040.jpg', 'train/ID364856705.jpg', 'train/ID423506847.jpg', 'train/ID4464212.jpg', 'train/ID482555369.jpg', 'train/ID490139972.jpg', 'train/ID520514019.jpg', 'train/ID528010569.jpg', 'train/ID529933668.jpg', 'train/ID550623196.jpg', 'train/ID560946727.jpg', 'train/ID567744300.jpg', 'train/ID576137678.jpg', 'train/ID638711343.jpg', 'train/ID656251220.jpg', 'train/ID668330410.jpg', 'train/ID686797154.jpg', 'train/ID706288721.jpg', 'train/ID72895391.jpg', 'train/ID742198710.jpg', 'train/ID750820644.jpg', 'train/ID751517087.jpg', 'train/ID755710743.jpg', 'train/ID786365141.jpg', 'train/ID797502182.jpg', 'train/ID799079114.jpg', 'train/ID808079729.jpg', 'train/ID885388135.jpg', 'train/ID896386823.jpg', 'train/ID94564238.jpg', 'train/ID969218269.jpg']
    
    target_dir = "./copied_dataset"
    
    copy_images(my_images, target_dir)