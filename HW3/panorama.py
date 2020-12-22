import numpy as np
from numpy.linalg import det, lstsq, norm
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
from functools import cmp_to_key
import multiprocessing as mp

class Panorama:
    def __init__(self, folder_list):
        input_list = open(folder_list, 'r').readlines()
        self.folder_list = [folder_name.strip() for folder_name in input_list]
        self.all_images, self.all_images_count, self.dataset_name = self.readInput()

        self.panorama()

    def readInput(self):
        all_images = []
        all_images_count = []
        dataset_name = []
        for directory_name in self.folder_list:
            current_dataset = []
            filenumber = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))])
            for i in range(1, filenumber+1):
                image = cv2.imread(directory_name + "/" + str(i) + ".jpg")
      
                image = self.cylindricalWarp(image)
                image = self.crop_left_right(image)
                
                current_dataset.append(image)

            all_images.append(current_dataset)
            all_images_count.append(filenumber)
            dataset_name.append(directory_name)
        return all_images, all_images_count, dataset_name

    def cylindricalWarp(self, img):
        focal_length = 700
        height, width, _ = img.shape
        cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)
        
        for y in range(-int(height/2), int(height/2)):
            for x in range(-int(width/2), int(width/2)):
                cylinder_x = focal_length*math.atan(x/focal_length)
                cylinder_y = focal_length*y/math.sqrt(x**2+focal_length**2)
                
                cylinder_x = round(cylinder_x + width/2)
                cylinder_y = round(cylinder_y + height/2)

                if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                    cylinder_proj[cylinder_y][cylinder_x] = img[y+int(height/2)][x+int(width/2)]
            
        return cylinder_proj

    def panorama(self):
        for folder_idx, folder_name in enumerate(tqdm(self.folder_list)):
            all_homography = []
            for query_idx in range(1, self.all_images_count[folder_idx]):
                train_idx = query_idx - 1
                self.keypoints_train, self.descriptors_train, self.keypoints_query, self.descriptors_query = self.SIFT(folder_idx, train_idx, query_idx)
                self.good = self.matchKeyPoints(folder_idx, train_idx, query_idx)
                self.homography = self.computeHomography()
                all_homography.append(self.homography)

            self.all_homography, self.all_projected_width, self.all_projected_height = self.computeOffsetHomography(folder_idx, all_homography)
            self.wrapImage(folder_idx)

    def gene_gaussian_kernel(self, sigma, num_intervals):
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

    def gen_gaussian_image(self, image, num_octaves, gaussian_kernels):
        gaussian_images = []
        for _ in range(num_octaves):
            gaussian_images_in_octave = [image]
            for gaussian_kernel in gaussian_kernels[1:]:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = cv2.resize(octave_base, (int(octave_base.shape[1]/2), int(octave_base.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
        return np.array(gaussian_images, dtype=object)

    def gen_DoG_images(self, gaussian_images):
        dog_images = []
        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for img_idx in range(1, len(gaussian_images_in_octave)):
                dog_images_in_octave.append(cv2.subtract(gaussian_images_in_octave[img_idx], gaussian_images_in_octave[img_idx-1]))
            dog_images.append(dog_images_in_octave)
        return np.array(dog_images, dtype=object)

    def sub_process_iterate_image_pixel(self, i_start_idx, j_start_idx, i_end_idx, j_end_idx, image_set, img_index, boarder_width, threshold, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, gaussian_images):
        keypoints = []
        for i in range(i_start_idx, i_end_idx):
            for j in range(j_start_idx, j_end_idx):
                if self.check_pixel_extreme(image_set[0][i-1:i+2, j-1:j+2], image_set[1][i-1:i+2, j-1:j+2], image_set[2][i-1:i+2, j-1:j+2], threshold):
                    localization_result = self.localize_extremum(i, j, img_index - 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, boarder_width)
                    if localization_result is not None:
                        keypoint, localized_image_index = localization_result
                        keypoints_with_orientations = self.compute_keypoint_orientation(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                        keypoints.extend(keypoints_with_orientations)
        return keypoints

    def process_findScaleSpaceExtrema(self, process_idx, gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
        threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
        keypoints = []

        dog_images_in_octave = dog_images[process_idx]
        octave_index = process_idx
        for image_index in range(2, len(dog_images_in_octave)):
            first_image = dog_images_in_octave[image_index-2]
            second_image = dog_images_in_octave[image_index-1]
            third_image = dog_images_in_octave[image_index]

            i_start_idx = image_border_width
            j_start_idx = image_border_width
            i_end_idx = first_image.shape[0] - image_border_width
            j_end_idx = first_image.shape[1] - image_border_width

            if first_image.shape[0] < 1000:
                sub_keypoints = self.sub_process_iterate_image_pixel(i_start_idx, j_start_idx, i_end_idx, j_end_idx, (first_image, second_image, third_image), image_index, image_border_width, threshold, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, gaussian_images)
                keypoints.extend(sub_keypoints)
            else:
                step_size = int(first_image.shape[0]/4)
                que = mp.Manager().Queue()
                process_list = []
                process_num = 4
                i_start_end_list = [i_start_idx, step_size, step_size*2, step_size*3, i_end_idx]

                for process_idx in range(process_num):
                    t = mp.Process(target=lambda q, arg1: q.put(self.sub_process_iterate_image_pixel(*arg1)), args=(que, (i_start_end_list[process_idx], j_start_idx, i_start_end_list[process_idx+1], j_end_idx, (first_image, second_image, third_image), image_index, image_border_width, threshold, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, gaussian_images)))
                    t.start()
                    process_list.append(t)
                for t in process_list:
                    t.join()
                while not que.empty():
                    keypoints.extend(que.get())

        return keypoints

    def check_pixel_extreme(self, first_subimage, second_subimage, third_subimage, threshold):
        center_pixel_value = second_subimage[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_subimage) and \
                    np.all(center_pixel_value >= third_subimage) and \
                    np.all(center_pixel_value >= second_subimage[0, :]) and \
                    np.all(center_pixel_value >= second_subimage[2, :]) and \
                    center_pixel_value >= second_subimage[1, 0] and \
                    center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_subimage) and \
                    np.all(center_pixel_value <= third_subimage) and \
                    np.all(center_pixel_value <= second_subimage[0, :]) and \
                    np.all(center_pixel_value <= second_subimage[2, :]) and \
                    center_pixel_value <= second_subimage[1, 0] and \
                    center_pixel_value <= second_subimage[1, 2]
        return False

    def localize_extremum(self, i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
            pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                                second_image[i-1:i+2, j-1:j+2],
                                third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.compute_gradient(pixel_cube)
            hessian = self.compute_hessian(pixel_cube)
            extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(np.round(extremum_update[0]))
            i += int(np.round(extremum_update[1]))
            image_index += int(np.round(extremum_update[2]))
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
                extremum_is_outside_image = True
                break

        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None

        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None

    def compute_gradient(self, pixel_array):
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def compute_hessian(self, pixel_array):
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])

    def compute_keypoint_orientation(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        radius = int(np.round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(np.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(np.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_index = int(np.round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < 1e-7:
                    orientation = 0
                new_keypoint = (keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    def compare_keypoints(self, keypoint1, keypoint2):
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def remove_duplicate_keypoints(self, keypoints):
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compare_keypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints

    def convert_keypoints_to_input_image_size(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def unpack_octave(self, keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale

    def process_generateDescriptors(self, process_idx, thread_num, keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        descriptors = []
        
        half_wind_width = 0.5*window_width
        half_wind_width_mins5 = half_wind_width - 0.5 

        start_idx = process_idx*int(len(keypoints)/thread_num)
        end_idx = (process_idx+1)*int(len(keypoints)/thread_num)
        if process_idx == thread_num-1:
            end_idx = len(keypoints)

        for _, keypoint in list(enumerate(keypoints))[start_idx:end_idx]:
            octave, layer, scale = self.unpack_octave(keypoint)
            gaussian_image = gaussian_images[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / (half_wind_width**2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            hist_width = scale_multiplier * 0.25 * scale * keypoint.size
            half_width = int(np.round(hist_width * 1.4142135623730951 * (window_width + 1)))
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2))) 

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + half_wind_width_mins5
                    col_bin = (col_rot / hist_width) + half_wind_width_mins5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(np.round(point[1] + row))
                        window_col = int(np.round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten() 

            threshold = norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), 1e-7)
        
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return (process_idx, np.array(descriptors, dtype='float32'))

    def detectAndCompute(self, image, sigma=1.6, num_intervals=5, image_border_width=5):
        image = image.astype('float32')
        image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        base_image = cv2.GaussianBlur(image, (0, 0), sigmaX=1, sigmaY=1)

        num_octaves = int(np.round(np.log(min(base_image.shape)) / np.log(2) - 1))
        gaussian_kernels = self.gene_gaussian_kernel(sigma, num_intervals)
        gaussian_images = self.gen_gaussian_image(base_image, num_octaves, gaussian_kernels)
        dog_images = self.gen_DoG_images(gaussian_images)

        ## findScaleSpaceExtrema
        process_list = []
        que = mp.Manager().Queue()
        process_num = len(dog_images)
        for process_idx in range(process_num):
            t = mp.Process(target=lambda q, arg1: q.put(self.process_findScaleSpaceExtrema(*arg1)), args=(que, (process_idx, gaussian_images, dog_images, num_intervals, sigma, image_border_width)))
            t.start()
            process_list.append(t)
        for t in process_list:
            t.join()
        
        keypoints = []
        while not que.empty():
            for kt in que.get():
                keypoints.append(cv2.KeyPoint(*kt[0], kt[1], kt[2], kt[3], kt[4]))

        keypoints = self.remove_duplicate_keypoints(keypoints)
        keypoints = self.convert_keypoints_to_input_image_size(keypoints)
        
        ## generateDescriptors
        que = mp.Manager().Queue()
        process_list = []
        process_num = min(12, len(keypoints))
        for process_idx in range(process_num):
            t = mp.Process(target=lambda q, arg1: q.put(self.process_generateDescriptors(*arg1)), args=(que, (process_idx, process_num, keypoints, gaussian_images)))
            t.start()
            process_list.append(t)
        for t in process_list:
            t.join()
        
        des_list = []
        descriptors = []
        while not que.empty():
            des_list.append(que.get())
        des_list = sorted(des_list, key=lambda x: x[0])
        for des in des_list:
            descriptors.append(des[1])
        descriptors = np.concatenate(descriptors, axis=0)

        return keypoints, descriptors

    def SIFT(self, folder_idx, train_idx, query_idx):
        # Sift for train image
        image_train = self.all_images[folder_idx][train_idx]
        gray_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2GRAY)
        sift_train = cv2.xfeatures2d.SIFT_create()
        keypoints_train, descriptors_train = self.detectAndCompute(gray_train)

        # Sift for query image
        image_query = self.all_images[folder_idx][query_idx]
        gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
        sift_query = cv2.xfeatures2d.SIFT_create()
        keypoints_query, descriptors_query = self.detectAndCompute(gray_query)

        return keypoints_train, descriptors_train, keypoints_query, descriptors_query

    def matchKeyPoints(self, folder_idx, train_idx, query_idx, threshold=0.75):
        # Match keypoints of all pictures pair

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.descriptors_query, self.descriptors_train, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append(m)

        return good

    def computeHomography(self):
        train_pts = np.float32([self.keypoints_train[m.trainIdx].pt for m in self.good]).reshape(-1, 1, 2)
        query_pts = np.float32([self.keypoints_query[m.queryIdx].pt for m in self.good]).reshape(-1, 1, 2)
        homography = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)[0]
        return homography

    def computeOffsetHomography(self, folder_idx, homography):
        aggregate_width = 0
        all_horizontal_homography = []
        all_projected_width = []

        aggregate_height = 0
        all_projected_height = []

        for query_idx in range(1, self.all_images_count[folder_idx]):
            train_idx = query_idx - 1
            query_image = self.all_images[folder_idx][query_idx].copy()
            h_query, w_query = query_image.shape[:2]

            # Move right images to right of the left images
            translation_matrix = np.array([[1, 0, aggregate_width], [0, 1, aggregate_height], [0, 0, 1]])
            homography[train_idx] = translation_matrix.dot(homography[train_idx])
            all_horizontal_homography.append(homography[train_idx])

            points_query = np.float32([[0, 0], [0, h_query], [w_query, h_query], [w_query, 0]]).reshape(-1, 1, 2)
            points_query_to_train = cv2.perspectiveTransform(points_query, homography[train_idx])

            [x_min, y_min] = np.int32(points_query_to_train.min(axis=0).ravel())
            [x_max, y_max] = np.int32(points_query_to_train.max(axis=0).ravel())
            width = int(round(x_max - x_min))
            height = int(round(y_max - y_min))

            aggregate_width += width
            aggregate_height += height
            all_projected_width.append(width)
            all_projected_height.append(height)

        return all_horizontal_homography, all_projected_width, all_projected_height
    
    def wrapImage(self, folder_idx):
        stitch_image = self.all_images[folder_idx][0].copy()

        aggregate_height, aggregate_width = 0, 0
        aggregate_overlap_width, aggregate_overlap_height = 0, 0
        
        for query_idx in tqdm(range(1, self.all_images_count[folder_idx])):
            train_idx = query_idx - 1
            train_image = self.all_images[folder_idx][train_idx].copy()
            query_image = self.all_images[folder_idx][query_idx].copy()
            h_query, w_query = query_image.shape[:2]
            h_train, w_train = stitch_image.shape[:2]

            # Move right images to left because overlap region of left images
            aggregate_overlap_matrix = np.array([[1, 0, aggregate_overlap_width], [0, 1, aggregate_overlap_height], [0, 0, 1]])
            self.all_homography[train_idx] = aggregate_overlap_matrix.dot(self.all_homography[train_idx])

            points_train = np.float32([[0, 0], [0, h_train], [w_train, h_train], [w_train, 0]]).reshape(-1, 1, 2)
            points_query = np.float32([[0, 0], [0, h_query], [w_query, h_query], [w_query, 0]]).reshape(-1, 1, 2)

            points_query_to_train = cv2.perspectiveTransform(points_query, self.all_homography[train_idx])
            points = np.concatenate((points_train, points_query_to_train), axis=0)

            [query_x_min, query_y_min] = np.int32(points_query_to_train.min(axis=0).ravel())
            [query_x_max, query_y_max] = np.int32(points_query_to_train.max(axis=0).ravel())
            
            [x_min, y_min] = np.int32(points.min(axis=0).ravel())
            [x_max, y_max] = np.int32(points.max(axis=0).ravel())

            new_size = (int(round(x_max - x_min)), int(round(y_max - y_min)))

            aggregate_width -= x_min
            aggregate_height -= y_min

            translation_matrix = np.array([[1, 0, 0], [0, 1, -y_min], [0, 0, 1]])
            translation_homography = translation_matrix.dot(self.all_homography[train_idx])

            current_stitch_image = cv2.warpPerspective(query_image, translation_homography, new_size)

            current_stitch_image[-y_min:h_train + -y_min, -x_min:w_train + -x_min] = stitch_image.copy()

            overlap_width = w_train + self.all_projected_width[train_idx] - current_stitch_image.shape[1]
            aggregate_overlap_width -= overlap_width

            overlap_height = h_train + self.all_projected_height[train_idx] - current_stitch_image.shape[0]
            aggregate_overlap_height -= overlap_height

            stitch_image = current_stitch_image.copy()

        # Remove black border
        stitch_image = self.crop_top_down(stitch_image)

        save_name = './' + self.dataset_name[folder_idx] + '.jpg'
        self.saveImage(stitch_image, save_name)

    def crop_top_down(self, img):
        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        upper, lower = [-1, -1]

        black_pixel_num_threshold_height = img.shape[1] // 30

        for y in range(thresh.shape[0]):
            if len(np.where(thresh[y, :] == 0)[0]) < black_pixel_num_threshold_height:
                upper = y
                break
            
        for y in range(thresh.shape[0]-1, 0, -1):
            if len(np.where(thresh[y, :] == 0)[0]) < black_pixel_num_threshold_height:
                lower = y
                break

        if upper == -1 and lower == -1:
            return img
        else:
            return img[upper:lower, :]

        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

        righter = -1
        black_pixel_num_threshold_width = img.shape[0] // 20
            
        for x in range(thresh.shape[1]-1, 0, -1):
            if len(np.where(thresh[:, x] == 0)[0]) < black_pixel_num_threshold_width:
                righter = x
                break

        if righter == -1:
            return img
        else:
            return img[:, :righter]

    def crop_left_right(self, img):
        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

        black_pixel_num_threshold_width = img.shape[0] // 30
        lefter, righter = [-1, -1]

        for x in range(thresh.shape[1]):
            if len(np.where(thresh[:, x] == 0)[0]) < black_pixel_num_threshold_width:
                lefter = x
                break
            
        for x in range(thresh.shape[1]-1, 0, -1):
            if len(np.where(thresh[:, x] == 0)[0]) < black_pixel_num_threshold_width:
                righter = x
                break

        
        if lefter == -1 and righter == -1:
            return img
        else:
            return img[:, lefter:righter]

    def saveImage(self, image, save_name):
        cv2.imwrite(save_name, image)


if __name__ == "__main__":
    panorama = Panorama('testfile.txt')