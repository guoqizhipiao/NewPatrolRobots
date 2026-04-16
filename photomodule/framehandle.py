from PIL import Image
import cv2
import numpy as np

class frame_handle():
    
    def cv2_to_pillow(self, image):
        """
        将OpenCV图像转换为Pillow图像 image为OpenCV对象
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image

    def pillow_to_cv2(self, image):
        """
        将Pillow图像转换为OpenCV图像 image为Pillow对象
        """
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def add_content_to_image(self, base_image, add_image, position=(0, 0), size=1):
        """
        在图像上添加内容 base_image/add_image为Pillow对象 psition为元组 size为数字0~1
        """
        add_image_width, add_image_height = add_image.size
        size = (int(add_image_width*size), int(add_image_height*size))

        add_image = add_image.resize(size)
        base_image.paste(add_image, position, mask=add_image)
        return base_image

    
