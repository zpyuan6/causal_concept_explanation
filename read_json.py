from PIL import Image
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    # img = Image.open("img\ADE_train_00010721_part_2.png")
    # plt.imshow(img)
    # plt.show()

    a = torch.rand(2,2)
    b = torch.rand(2,2)

    print(a)
    print(b)
    print(a@b)
    print(torch.matmul(a,b))