import os
import shutil

new_raw_path = r"new_raw"
new_masks_path = r"new_masks"
matched_masks_path = r"matches/matched_masks"
matched_pics_path = r"matches/matched_raws"

folder_new_masks = os.listdir(new_masks_path)


# uses entire dataset to check for matches if true, if false uses imgs contained in "new_masks"
entire_dataset = False


if entire_dataset == True:
    # path to imgs folder that contains entire dataset (incase its needed):
    new_raw_path = r"C:\Users\veyse\sciebo\BA_Projekt\Project\DropletsMask\raw"
    folder_all_imgs = os.listdir(new_raw_path)
else:
    new_raw_path = r"new_raw"
    folder_all_imgs = os.listdir(new_raw_path)



#copy all the new masks with the same name if they exist as img into two new folders
for img in folder_all_imgs:
    for mask in folder_new_masks:
        if img == mask:
            shutil.copy(os.path.join(new_masks_path,mask),matched_masks_path)
            shutil.copy(os.path.join(new_raw_path, img), matched_pics_path)

"""check which images are not in the masks folder"""
"""for img in folder_all_imgs:
    if img not in folder_new_masks:
        print(img)"""


"""get img number"""
#(folder_all_imgs[102])



