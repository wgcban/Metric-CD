import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.linear_model import LinearRegression

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))

def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m

def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result
    
tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

# output_t = tf(torch.squeeze(output_t.cpu()))
# output_t = output_t.squeeze().cpu().numpy()
# output_t = output_t.transpose((1, 2, 0))
# m_t = utls.get_mapping_func(image_resized, output_t)
# output_t = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_t))