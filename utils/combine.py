import math
from PIL import Image

column = 1
width = 1250
height = 1250
size = (width, height)

def combine2pic(root, id):
    list_im = [root + 'image/Iutput' + str(id) + '.jpg', root + 'image/Output' + str(id) + '.jpg']
    imgs = [Image.open(i) for i in list_im]

    row_num = math.ceil(len(imgs) / column)
    target = Image.new('RGB', (width * column, height * row_num))
    for i in range(len(list_im)):
        if i % column == 0:
            end = len(list_im) if i + column > len(list_im) else i + column
            for col, image in enumerate(imgs[i:i + column]):
                target.paste(image, (width * col, height * (i // column)))
    # target.show()
    target = target.resize((target.size[0], target.size[1]//2)).convert('L')

    target.save(root + 'train/c2img/cbimg' + str(id) + '.jpg')


if __name__ == '__main__':
    root = '/home/huxleyhu/baidunetdiskdownload/深度生成模型DeepGenerativeModel/LO_Temperature_Problem1/'
    for i in range(1, 40001):
        combine2pic(root, i)