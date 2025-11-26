# 该脚本是优化之后的imgtrans.py 速度更快

import random
import cv2
import numpy as np


# crop_size: 112
# scale: 0.0078125
# mean_value: 127.5
# mean_value: 127.5
# mean_value: 127.5
# augmention: true
# angle: 60
# scale_max: 1.2
# scale_min: 0.7
# random_crop: true
# factor: 3
# quality: 100
# noise_num: 200
# brightness_prob: 0.5
# brightness_delta: 15
# contrast_prob: 0.2
# contrast_lower: 0.5
# contrast_upper: 1.5
# hue_prob: 0.2
# hue_delta: 18
# saturation_prob: 0.2
# saturation_lower: 0.5
# saturation_upper: 1.5
# random_order_prob: 0


# crop_size: 128
# augmention: true
# angle: 60
# scale_max: 1.2
# scale_min: 0.8
# random_crop: true
# factor: 2.5
# quality: 100
# noise_num: 300
# brightness_prob: 0.5
# brightness_delta: 15
# contrast_prob: 0.2
# contrast_lower: 0.5
# contrast_upper: 1.5
# hue_prob: 0.2
# hue_delta: 18
# saturation_prob: 0.2
# saturation_lower: 0.5
# saturation_upper: 1.5
# random_order_prob: 0

# 预计算gama变化所需要的参数，防止在每一张图像进行gama调整的时候都要循环生成table
# 是根据原来设置的最大最小值定制的，如果最大最小值修改了，也要做相应的修改
# def compute_gama_table():
#     gama_list = {'2.0': np.linspace(0.5, 2.0, 150), '5.0': np.linspace(0.5, 5.0, 450)}  #
#     table_list = {'2.0': {}, '5.0': {}}
#     for num in (2.0, 5.0):
#         for gama in gama_list[str(num)]:
#             invGamma = 1.0 / gama
#             table_list[str(num)][str(gama)] = np.array(
#                 [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return gama_list, table_list


class Transform():
    '''
    distor 常规增广
    img 减均值
    '''

    # 将adjust_gama的计算值定义为静态变量，不用在每一次调用adjust_gama的时候重新计算
    # gama_list, table_list = compute_gama_table()

    def __init__(self, args=None, nonflip=False):
        trans_args = {
            'crop_size': 112,
            'augmention': True,
            'angle': 40,  # 随机旋转的角度,原数值：60
            'scale_max': 1.2,
            'scale_min': 0.7,
            'random_crop': True,
            'factor': 3,
            'quality': 100,
            'noise_num': 200,
        }
        dis_param = {
            'brightness_prob': 0.5,
            'brightness_delta': 15,
            'contrast_prob': 0.2,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 50,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            # 'random_order_prob': 0,
            'min_gama': 0.5,
            'max_gama': 5.0,
            'stride_gama': 0.01,
            'quality': 100,

        }
        # 默认参数
        self.__dict__.update(dis_param)
        if trans_args is not None:
            self.__dict__.update(trans_args)

        self.nonflip = nonflip


    def __call__(self, img):
        # 光亮
        img = self.RandomBrightness(img, self.brightness_prob, self.brightness_delta)  # 50%

        if random.uniform(0, 1.0) > 0.5:  # 0.5
            # probs = np.random.uniform(0.0, 0.01, size=9)
            probs = [random.uniform(0, 1.0) for _ in range(9)] # 这个速度更快一点
            # #灰度化
            if probs[0] < 0.4:  # 20%
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 发亮变暗
            if probs[1] < 0.12:  # 6%
                img = self.gama_com(img, 0.5, 5.0, 0.01)
            else:
                img = self.gama_com(img, 0.5, 2.0, 0.01)
            # 下面这个是使用预计算table的gama调整方法
            # if probs[1] < 0.12:  # 6%
            #     img = self.gama_com_test(img, 0.5, 5.0, 0.01)
            # else:
            #     img = self.gama_com_test(img, 0.5, 2.0, 0.01)

            # from PIL import Image
            # img_pil = Image.fromarray(img, mode='RGB')  # RGB 模式
            # img_pil.show()
            # 颜色
            if probs[2] < 0.4:  # 20%
                img = self.DistortImage(img)

            # jpg图像压缩
            if probs[3] < 0.2:  # 0.2 10%
                prob = random.uniform(0, 1.0)
                quality = (100 - 30) * prob + 30
                ret, img_encode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                img = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)

            # 随机三种高斯模糊变化
            if probs[4] < 0.2:  # 10%
                kernel = random.choice([1, 3, 5])
                # img1 = cv2.GaussianBlur(img, (kernel, kernel), 1.5)  # 高斯核越大越模糊
                img = self.fast_gaussian_blur(img, kernel, sigma=1.5)  # 这个速度更快

            # # 图像腐蚀
            if probs[5] < 0.01:  # 0.5%
                kernel = cv2.getStructuringElement(2, (3, 3), (1, 1))
                # 创建形态学操作的结构元素 核的尺寸必须为奇数！！！
                img = cv2.erode(img, kernel, iterations=1)

            # 图像膨胀
            if probs[6] < 0.01:  # 0.5%
                kernel = cv2.getStructuringElement(2, (3, 3), (1, 1))
                img = cv2.dilate(img, kernel, iterations=1)

            # 加噪  i行j列
            if probs[7] < 0.2:  # 10%
                # 下面这是逐个像素的加噪点，速度太慢了，可以选择生成带有噪点的mask，直接贴到图片上，并且使用numpy的random更快
                # for k in range(self.noise_num):
                #     b = random.randint(0, 254)
                #     pixel = random.choice([[255,255,255],
                #                           [0, 0, 0],
                #                           [b, b, b]])
                #     i = random.randint(0, img.shape[0]-1)
                #     j = random.randint(0, img.shape[1]-1)
                #     img[i,j] = pixel

                # 生成self.noise_num 个随机噪点的位置
                i = np.random.randint(0, img.shape[0] - 1, size=self.noise_num)
                j = np.random.randint(0, img.shape[1] - 1, size=self.noise_num)
                # # 生成像素
                pixel_fixed = np.array([[255, 255, 255], [0, 0, 0]])
                pixel_random = np.random.randint(0, 254, size=(self.noise_num, 1))
                pixel_random = np.repeat(pixel_random, 3, axis=1)
                choice = np.random.randint(0, 3, size=self.noise_num)
                pixels = np.zeros((self.noise_num, 3), dtype=np.uint8)
                pixels[choice < 2] = pixel_fixed[choice[choice < 2]]
                pixels[choice == 2] = pixel_random[choice == 2]
                img[i, j] = pixels
                del pixels

            if img.shape[0] < self.crop_size or \
                    img.shape[1] < self.crop_size:
                img = cv2.resize(img, (self.crop_size, self.crop_size))

            # 随机旋转，尺度
            if probs[8] < 0.4:  # 20%的概率
                img = self.augmentation_rotate_scale(img)

        if img.shape[0] < self.crop_size or \
                img.shape[1] < self.crop_size:
            img = cv2.resize(img, (self.crop_size, self.crop_size))

        # 随机裁剪
        # 对齐后图片不做裁剪，非对齐图片裁剪
        img = self.augmentation_croppad(img)  # 50%的概率会进行裁剪

        if not self.nonflip:  # 这个目前是不采用的
            img = self.augmentation_flip(img)

        return img

    def fast_gaussian_blur(self,img, kernel_size=3, sigma=1.5):
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
        return cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)
    # 下面这两个函数是预先计算 gama调整所需要的table，能起到一定的提速效果

    def adjust_gama_test(self, img, max_gama, gama):
        table = Transform.table_list[str(max_gama)][str(gama)]
        return cv2.LUT(img, table)

    def gama_com_test(self, img, min_gama, max_gama, stride_gama):
        list1 = Transform.gama_list[str(max_gama)]
        random_num = int(random.uniform(0, len(list1)))
        gama = list1[random_num]
        img = self.adjust_gama_test(img, max_gama, gama)
        return img

    def DistortImage(self, img):
        prob = random.uniform(0, 1.0)
        if (prob > 0.5):
            # 对比度，饱和，色调
            img = self.RandomContrast(img, self.contrast_prob, self.contrast_lower, self.contrast_upper)
            img = self.RandomSaturation(img, self.saturation_prob, self.saturation_lower, self.saturation_upper)
            img = self.RandomHue(img, self.hue_prob, self.hue_delta)
            # img = self.RandomOrderChannels(img, self.random_order_prob)
        else:
            img = self.RandomSaturation(img, self.saturation_prob, self.saturation_lower, self.saturation_upper)
            img = self.RandomHue(img, self.hue_prob, self.hue_delta)
            img = self.RandomContrast(img, self.contrast_prob, self.contrast_lower, self.contrast_upper)
            # img = self.RandomOrderChannels(img, self.random_order_prob)
        return img

    # 随机亮度
    def RandomBrightness(self, img, brightness_prob, brightness_delta):
        prob = random.uniform(0, 1.0)
        if (prob < brightness_prob):
            img = img.astype(np.float)
            assert brightness_delta >= 0
            delta = random.uniform(-brightness_delta, brightness_delta)
            img += delta  # 每一个像素值随机加上一个偏移值，导致图像变暗或者变亮
            img = np.maximum(img, 0)
            img = np.minimum(img, 255)
            img = img.astype(np.uint8)
        return img

    # 随机对比度
    def RandomContrast(self, img, contrast_prob, contrast_lower, contrast_upper):
        prob = random.uniform(0, 1.0)
        if (prob < contrast_prob):
            img = img.astype(np.float)
            upper = contrast_upper
            lower = contrast_lower
            assert upper >= lower
            assert lower >= 0
            delta = random.uniform(lower, upper)
            if abs(delta - 1.0) > 1e-3:
                img *= delta
            img = np.maximum(img, 0)
            img = np.minimum(img, 255)
            img = img.astype(np.uint8)
        return img

    # 随机饱和度
    def RandomSaturation(self, img, saturation_prob, saturation_lower, saturation_upper):
        prob = random.uniform(0, 1.0)
        if (prob < saturation_prob):
            img = img.astype(np.float32)
            upper = saturation_upper
            lower = saturation_lower
            assert upper >= lower
            assert lower >= 0
            delta = random.uniform(lower, upper)
            if abs(delta - 1.0) > 1e-3:
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_hsv[:, :, 1] *= delta
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            img = np.maximum(img, 0)
            img = np.minimum(img, 255)
            img = img.astype(np.uint8)
        return img

    # 随机色调
    def RandomHue(self, img, hue_prob, hue_delta):
        prob = random.uniform(0, 1.0)
        if (prob < hue_prob):
            img = img.astype(np.float32)
            assert hue_delta >= 0
            delta = random.uniform(-hue_delta, hue_delta)
            if abs(delta) > 0:
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_hsv[:, :, 0] += delta
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            img = np.maximum(img, 0)
            img = np.minimum(img, 255)
            img = img.astype(np.uint8)
        return img

    # 随机rgb的channel
    def RandomOrderChannels(self, img, random_order_prob):
        prob = random.uniform(0, 1.0)
        if (prob < random_order_prob):
            img = img.astype(np.float32)
            b, g, r = cv2.split(img)
            channel = [b, g, r]
            np.random.shuffle(channel)
            b = channel[0]
            g = channel[1]
            r = channel[2]
            img = cv2.merge((b, g, r))
            img = np.maximum(img, 0)
            img = np.minimum(img, 255)
            img = img.astype(np.uint8)
        return img

    # 随机旋转
    def augmentation_rotate_scale(self, img):
        '''
        最大旋转角度 angle
        尺度缩放最大最小值 scale_max,scale_min
        '''
        assert self.angle < 90
        assert self.scale_max < 1.3
        assert self.scale_min > 0.4

        # 随机旋转尺度变换
        degree_scale = random.uniform(0, 1.0)
        dice_scale = random.uniform(0, 1.0)
        degree = self.angle * degree_scale
        scale = (self.scale_max - self.scale_min) * dice_scale + self.scale_min
        # 生成仿射变换矩阵 以图像中心为旋转中心
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), degree, scale)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        # flag 为插值方法，borderMode为超出边界要处理的方式     borderValue为当 borderMode = BORDER_CONSTANT 时，填充值，默认是黑色（0）
        return img

    # 随机裁剪
    def augmentation_croppad(self, img):
        '''
        裁减尺寸 crop_size
        是否随机裁剪 random_crop
        裁剪因子 factor
        '''
        assert self.crop_size <= img.shape[0]
        assert self.crop_size <= img.shape[1]
        assert self.factor <= 3.0

        # 在 + - crop_size/(2*factor) 范围内做裁剪增广
        dice_x = random.uniform(0, 1.0)
        dice_y = random.uniform(0, 1.0)
        # 下面是x,y方向上的随机偏移量
        tr_x = (self.crop_size / self.factor) * dice_x - self.crop_size / self.factor / 2.0
        tr_y = (self.crop_size / self.factor) * dice_y - self.crop_size / self.factor / 2.0

        if self.random_crop and random.uniform(0, 0.1) < 0.5:  # 50%的概率进行裁剪
            img = cv2.resize(img, (self.crop_size, self.crop_size))

            srcTri = np.float32([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]])
            dstTri = np.float32([[tr_x, tr_y], [img.shape[1] - 1 + tr_x, tr_y], [tr_x, img.shape[0] - 1 + tr_y]])

            M = cv2.getAffineTransform(srcTri, dstTri) # 获取仿射变换矩阵

            img = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        else:
            img = cv2.resize(img, (self.crop_size, self.crop_size))

        return img

    # 左右翻转
    def augmentation_flip(self, img):
        if random.uniform(0, 1.0) < 0.5:
            img = cv2.flip(img, 1)
        return img

    def adjust_gama(self, img, gama):
        invGamma = 1.0 / gama
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def gama_com(self, img, min_gama, max_gama, stride_gama):
        num = (max_gama - min_gama) / stride_gama
        list1 = np.linspace(min_gama, max_gama, int(num))
        random_num = int(random.uniform(0, len(list1)))
        gama = list1[random_num]
        img = self.adjust_gama(img, gama)
        return img





if __name__ == '__main__':
    from dataset_noalign import MXFaceDataset

    # MX = MXFaceDataset("test_tmps/test", local_rank=1)
    # for i in MX:
    #     sample,label = i
    #     imgtrans = Transform()
    #     img = imgtrans(sample)
    #
    #     cv2.imshow('a', img[:,:,::-1])
    #     k = cv2.waitKey(-1)
    #     if k==ord("q"):
    #         break

    #
    #
    # import cv2
    #
    # img_path = "/home/laona/test"
    # for i in os.listdir(img_path):
    #     img = os.path.join(img_path,i)
    #     face = cv2.imread(img)
    #
    #     imgtrans = Transform()
    #     face1 = imgtrans(face)
    #
    #     #
    #     # cv2.imshow('a', face1)
    #     # k = cv2.waitKey(-1)
    #     # if k==ord("q"):
    #     #     break
    #
    #     face2 = cv2.resize(face, (400, 400))
    #     face3 = cv2.resize(face1, (400, 400))
    #     imgs = np.hstack((face2, face3))
    #
    #     cv2.namedWindow("mutil_pic")
    #     cv2.imshow("mutil_pic", imgs)
    #     k = cv2.waitKey(-1)
    #     if k == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
    #
    #
    #
