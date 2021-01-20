import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from read_MODIS_03 import get_solarZenith
from read_MODIS_02 import prepare_data

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).

    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.

    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    Examples
    --------
    >>> img = array([[ 91.06794177,   3.39058326,  84.4221549 ],
                     [ 73.88003259,  80.91433048,   4.88878881],
                     [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)

    """

    # if data.dtype == np.uint8:
    #     return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = np.nanmin(data)
    if cmax is None:
        cmax = np.nanmax(data)

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return (bytedata) + (low)

def get_enhanced_RGB(RGB):
    def scale_image(image):
        along_track = image.shape[0]
        cross_track = image.shape[1]

        x = np.array([0,  30,  60, 120, 190, 255])
        y = np.array([0, 110, 160, 210, 240, 255])

        scaled = np.zeros((along_track, cross_track))
        for i in range(len(x)-1):
            x1 = x[i]
            x2 = x[i+1]
            y1 = y[i]
            y2 = y[i+1]
            m = (y2 - y1) / float(x2 - x1)
            b = y2 - (m *x2)
            mask = ((image >= x1) & (image < x2))
            scaled = scaled + mask * np.asarray(m * image + b)

        mask = image >= x2
        scaled = scaled + (mask * 255)
        return scaled


    # case = ['/home/javi/MODIS_Training/BRF_RGB_Toronto.npz',
    #         '/home/javi/MODIS_Training/BRF_RGB_Aerosol.npz' ]
    # rgb = np.load(case[1])['arr_0'][:]
    enhanced_RGB = np.zeros_like(RGB)
    for i in range(3):
        enhanced_RGB[:, :, i] = scale_image(bytescale(RGB[:, :, i]))

    return enhanced_RGB

def get_BRF_RGB(filename_MOD_02, filename_MOD_03, path):

    #get Ref RGB to compare by eye
    cos_sza = np.cos(np.deg2rad(get_solarZenith(path + filename_MOD_03)))
    fieldnames_list  = ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB']
    rad_or_ref = False #True for radiance, False for reflectance
    #make channels for RGB photo (index 01234 -> band 34567)
    image_blue  = prepare_data(path + filename_MOD_02, fieldnames_list[0],rad_or_ref)[0,:,:] #band 3 from 500 meter res
    image_green = prepare_data(path + filename_MOD_02, fieldnames_list[0],rad_or_ref)[1,:,:] #band 4 from 500 meter res
    image_red   = prepare_data(path + filename_MOD_02, fieldnames_list[1],rad_or_ref)[0,:,:] #band 1 from 250 meter res
    #convert to BRF
    image_blue  /= cos_sza
    image_green /= cos_sza
    image_red   /= cos_sza

    RGB = np.dstack((image_red, image_green, image_blue))

    return RGB

if __name__ == '__main__':

    import matplotlib.pyplot as plt


    filename_MOD_03 ='/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD03.A2018248.1630.061.2018248230625.hdf'
    filename_MOD_02 = '/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD021KM.A2018248.1630.061.2018249014250.hdf'
    path = ''
    RGB = get_BRF_RGB(filename_MOD_02, filename_MOD_03, path)
    RGB_enhanced  = get_enhanced_RGB(RGB)

    plt.imshow(RGB_enhanced)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./clouds_over_ice.png', dpi=300)
    plt.show()
