# from imagededup.methods import PHash
# from imagededup.utils import plot_duplicates
# phasher = PHash()

# encodings = phasher.encode_images(image_dir='/home/vishal/datasets/vishal_recheck/4_images/')
# duplicates = phasher.find_duplicates(encoding_map=encodings)

# for img_name in duplicates:
#     plot_duplicates(image_dir='/home/vishal/datasets/vishal_recheck/4_images/',
#                 duplicate_map=duplicates,
#                 filename=img_name)

# from imagededup.methods import PHash
# myencoder = PHash()
# duplicates = myencoder.find_duplicates(image_dir='/home/vishal/datasets/vishal_recheck/4_images/', max_distance_threshold=15, scores=True,
# outfile='results.json')




# from imagededup.methods import PHash
# whasher = PHash()
# files_to_remove = whasher.find_duplicates_to_remove(image_dir='/home/vishal/datasets/vishal_recheck/4_images/', max_distance_threshold=15)
# # print(len(files_to_remove))
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from imagededup.methods import CNN
# myencoder = CNN()
# duplicates = myencoder.find_duplicates(image_dir='/home/vishal/datasets/vishal_recheck/4_images/', min_similarity_threshold=0.85, scores=True,
# outfile='results.json')

desti_dupli = "/home/vishal/datasets/vishal_recheck/4_dupli"
desti_clean = "/home/vishal/datasets/vishal_recheck/4_clean"


base = "/home/vishal/datasets/vishal_recheck/4_images/"

from imagededup.methods import CNN
import glob
import os
import shutil
myencoder = CNN()
duplicates = myencoder.find_duplicates_to_remove(image_dir='/home/vishal/datasets/vishal_recheck/4_images/' , min_similarity_threshold=.98)
print(len(duplicates), type(duplicates))

for img in glob.glob('/home/vishal/datasets/vishal_recheck/4_images/'+"*.jpg"):
    if os.path.basename(img) not in duplicates:
        print(img)
        shutil.copy(img, desti_clean)
    else:
        shutil.copy(img, desti_dupli)

# from imagededup.methods import CNN

# cnn = CNN()
# encodings = cnn.encode_images(image_dir="/home/vishal/datasets/vishal_recheck/4_images/")
# duplicates = cnn.find_duplicates(encoding_map=encodings)
# print(duplicates)