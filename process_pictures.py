# import cv2
#
# def auto_crop_center(image_path, target_width, target_height):
#     original_image = cv2.imread(image_path)
#     original_height, original_width, _ = original_image.shape
#
#     left = (original_width - target_width) // 2
#     top = (original_height - target_height) // 2
#     right = (original_width + target_width) // 2
#     bottom = (original_height + target_height) // 2
#
#     cropped_image = original_image[top:bottom, left:right]
#     cv2.imwrite("cropped_image.jpg", cropped_image)
#
# # 替换以下参数为您的实际情况
# image_path = "D:\\input\\my\\2.png"
# target_width = 800
# target_height = 600
#
# auto_crop_center(image_path, target_width, target_height)
