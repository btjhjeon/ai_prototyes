import os
import shutil


if __name__ == "__main__":
    root = 'QA_images/human_evaluation'
    output = 'QA_images/human_evaluation/all'
    dirlist = os.listdir(root)
    os.makedirs(output)
    for alphabet in dirlist:
        for file in os.listdir(os.path.join(root, alphabet)):
            file_name, file_ext = os.path.splitext(file)
            src_path = os.path.join(root, alphabet, file)
            dst_path = os.path.join(output, f'{file_name}_{alphabet}{file_ext}')
            shutil.copy(src_path, dst_path)
