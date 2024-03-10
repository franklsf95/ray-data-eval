import unittest
from PIL import Image
import timeit

PATH = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_18.JPEG"

DEFAULT_IMAGE_SIZE = 224
NUM_TRIALS = 10000


class TestImageResize(unittest.TestCase):
    """
    Load and resize a single image, repeated NUM_TRIALS times, to measure 
    and compare the performance of default (Image.BICUBIC) vs. proposed (Image.BILINEAR) 
    resampling modes for PIL resize.

    
    Example Output:
    Time taken with default (BICUBIC) implementation: 45.74734696099995
    Time taken with BILINEAR mode: 40.40636501500012
    .
    ----------------------------------------------------------------------
    Ran 1 test in 86.155s

    OK
    """

    def test_resize_image(self):
        def resize_image_default():
            """
            Default, i.e. BICUBIC resampling mode
            """
            with Image.open(PATH) as img:
                img.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

        def resize_image_bilinear():
            with Image.open(PATH) as img:
                img.resize(
                    (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), resample=Image.BILINEAR
                )

        time_default = timeit.timeit(resize_image_default, number=NUM_TRIALS)
        time_bilinear = timeit.timeit(resize_image_bilinear, number=NUM_TRIALS)

        print("Time taken with default (BICUBIC) implementation:", time_default)
        print("Time taken with proposed (BILINEAR) mode:", time_bilinear)


if __name__ == "__main__":
    unittest.main()
