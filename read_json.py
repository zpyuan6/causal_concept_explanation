from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = Image.open("img\ADE_train_00010721_part_2.png")
    plt.imshow(img)
    plt.show()
