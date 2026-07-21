import os
import cv2
import glob

original_format_folder = "../data/uavid_val"
out_folder = "../data/flat_uavid_val"
windows = False

folders = ["Images", "Labels"]

##TODO (by you)
### APPLY THIS SCRIPT ON uavid_train, uavid_val, uavid_test

def flatten():
    for i in range(len(folders)):
        file_list = sorted(
            glob.glob(os.path.join(original_format_folder, '*/' + folders[i] + '/*.png')))  # Images,Labels
        if windows:
            file_list = [file.replace("\\", "/") for file in file_list]

        for image_name in file_list:
            img = cv2.imread(image_name)
            splited_name = image_name.split("/")

            seq = int(splited_name[-3].replace("seq", ""))
            number = int(splited_name[-1].replace(".png", ""))

            image_id = (seq * 1000 + number)

            full_folder = '/'.join(os.path.join(out_folder, folders[i] + "/" + str(image_id) + ".png").split('/')[:-1])

            if not os.path.exists(full_folder):
                os.makedirs(full_folder)

            cv2.imwrite(os.path.join(out_folder, folders[i] + "/" + str(image_id) + ".png"), img)
            print(i, " - done with img -", image_id)
            print(os.path.join(out_folder, folders[i] + "/" + str(image_id) + ".png"))


if __name__ == "__main__":
    flatten()
