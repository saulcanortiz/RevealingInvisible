import cv2
import numpy as np
import torch
import os
from parametros import *
import math
import random


class NetDataset(torch.utils.data.Dataset):



    def __init__(self, input_path, groundtruth_path=None):
        self.input_path = input_path
        self.groundtruth = groundtruth_path
        self.files = []
        for filename in os.listdir(input_path):
            '''
            if os.path.isfile(os.path.join(input_path, filename)) and \
               ((groundtruth_path == None) or \
               os.path.isfile(os.path.join(groundtruth_path, filename))):
            '''
            print('Adding file: {}'.format(filename))
            self.files.append(filename)
        print('Total files in dataset: {}'.format(len(self.files)))


    def _magnificate(self,patch_y0, patch_x0, patch_x, patch_y, input_img, patch, minpix, maxpix, gauss_border):

        pixel_x_plus = random.randint(minpix, maxpix)
        pixel_y_plus = pixel_x_plus

        patch_resized_x = patch_x + pixel_x_plus
        patch_resized_y = patch_y + pixel_y_plus
        dim = (patch_resized_y,patch_resized_x)
        patch_resized = cv2.resize(patch, dim, interpolation = cv2.INTER_CUBIC)

        k1 = math.log(gauss_border)/2


        im = input_img.copy()
        # im1 = patch_resized.copy()
        for k in range(3):
            for y in range(0, patch_resized_y):
                for x in range(0, patch_resized_x):
                    # print("patch_x0:", patch_x0,"patch_y0:", patch_y0)
                    # print("x:", x,"y:", y, "k:",k)
                    # print("input_img_x:", input_img.shape[1],"input_img_y:", input_img.shape[0])
                    if (patch_y0 + y) > im.shape[0]:
                        # print("IndexError")
                        # print("patch_x0:", patch_x0,"patch_y0:", patch_y0)
                        # print("x:", x,"y:", y, "k:",k)
                        # print("input_img_x:", input_img.shape[1],"input_img_y:", input_img.shape[0])
                        break

                    elif (patch_x0 + x) > im.shape[1]:
                        # print("IndexError")
                        # print("IndexError")
                        # print("patch_x0:", patch_x0,"patch_y0:", patch_y0)
                        # print("input_img_x:", input_img.shape[1],"input_img_y:", input_img.shape[0])
                        # print("x:", x,"y:", y, "k:",k)
                        break
                    else:
                        alpha = math.exp(k1*((x - patch_resized_x/2)**2/(patch_resized_x/2)**2 + (y - patch_resized_y/2)**2/(patch_resized_y/2)**2))
                        # im1[y,x,k] = alpha*patch_resized[y,x,k] #ver alpha blending
                        im[int(patch_y0 - pixel_y_plus/2 + y), int(patch_x0 - pixel_x_plus/2 + x), k] = (1-alpha)*(im[int(patch_y0 - pixel_y_plus/2 + y),int(patch_x0 - pixel_x_plus/2 + x), k]) + alpha*(patch_resized[y,x,k])
        
        # cv2.imshow("input",input_img)
        # cv2.waitKey(0)
        # cv2.imshow("patch",patch)
        # cv2.waitKey(0)
        # cv2.imshow("patch_resized",patch_resized)
        # cv2.waitKey(0)
        # cv2.imshow("alpha_blending",im1)
        # cv2.waitKey(0)
        # cv2.imshow("modified",im)
        # cv2.waitKey(0)

        return im


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]
        input_img = cv2.imread(os.path.join(self.input_path, filename), cv2.IMREAD_COLOR)

        input_img_y, input_img_x, channels = input_img.shape

        # maxpixv (tiene sentido?)
        # ¿patch cuadrado?
        # 
        patch_y0 = random.randint(0, input_img_y - (patch_y + maxpixv))
        patch_x0 = random.randint(0, input_img_x - (patch_x + maxpixv))
        
        patch_y1 = patch_y0 + patch_y
        patch_x1 = patch_x0 + patch_x
        
        patch = input_img[patch_y0:patch_y1,patch_x0:patch_x1].copy()


        img1 = self._magnificate(patch_y0, patch_x0, patch_x, patch_y, input_img, patch, minpix, maxpix, gaussborder)
        img2 = self._magnificate(patch_y0, patch_x0, patch_x, patch_y, input_img, patch, minpixv, maxpixv, gaussborderv)


        return [torch.cat((self.__img2tensor(input_img),self.__img2tensor(img1)),dim = 0), self.__img2tensor(img2)]

    def __img2tensor(self, img):
        img = img.astype(np.float32) / 255.0

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        channels =  img.shape[2]
        if channels == 1:
            img = img[:, :, [0]]
        elif channels == 3:
            img = img[:, :, [2, 1, 0]]
        elif channels == 4:
            img = img[:, :, [2, 1, 0, 3]]

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        return img

    def __tensor2img(self, tensor, min_max=(0., 1.)):
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        ndim = tensor.dim()
        if ndim == 4:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(np.math.sqrt(n_img)), normalize=False).numpy()
            if tensor.size()[1] == 3:
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif tensor.size()[1] == 4:
                img_np = np.transpose(img_np[[2, 1, 0, 3], :, :], (1, 2, 0))  # HWC, BGR

        elif ndim == 3:
            img_np = tensor.detach().numpy()
            # img_np = tensor.numpy()
            if tensor.size()[0] == 3:
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif tensor.size()[0] == 4:
                img_np = np.transpose(img_np[[2, 1, 0, 3], :, :], (1, 2, 0))  # HWC, BGR

        elif ndim == 2:
            img_np = tensor.numpy()

        else:
            raise TypeError(
                'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(ndim))

        img_np = (img_np * 255.0).round()
        return img_np.astype(np.uint8)


    def save_output(self, tensor, filename):
        img = self.__tensor2img(tensor)
        cv2.imwrite(filename, img)

    def get_filename(self, index):
        return self.files[index]
