from ImageProcessor import ImageProcessor

processor = ImageProcessor()
processor.load_image("database/images/01_dr.JPG", "database/manual/01_dr.tif")
processor.scale_image(1000, 1000)
# processor.show_rgb_spectrum()

# processor.pre_process_image()
# processor.show_image()
processor.erase_channel([0, 2])
processor.to_grey_scale()
processor.pre_process_image()
# processor.show_image()
