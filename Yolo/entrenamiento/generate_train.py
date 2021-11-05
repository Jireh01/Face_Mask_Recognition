import os

image_files = []
#os.chdir(os.path.join("/content/darknet/data/", "/content/darknet/obj/"))
os.chdir(os.path.join("darknet","data","colab"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("/content/darknet/data/colab/" + filename)
print(image_files)
os.chdir("/content/darknet/data/")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")