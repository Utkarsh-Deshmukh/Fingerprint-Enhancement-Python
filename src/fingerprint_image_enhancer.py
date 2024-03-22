# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4 19:46:32 2020

@author: utkarsh
"""

import math
import numpy as np
import cv2
from scipy import signal
from scipy import ndimage
import scipy


# pylint: disable=too-many-instance-attributes, too-many-function-args, too-many-locals
class FingerprintImageEnhancer:
    """Fingerprint Enhancer Object."""

    def __init__(self):
        """Initialize the object."""
        self.ridge_segment_blksze = 16
        self.ridge_segment_thresh = 0.1
        self.gradient_sigma = 1
        self.block_sigma = 7
        self.orient_smooth_sigma = 7
        self.ridge_freq_blksze = 38
        self.ridge_freq_windsze = 5
        self.min_wave_length = 5
        self.max_wave_length = 15
        self.relative_scale_factor_x = 0.65
        self.relative_scale_factor_y = 0.65
        self.angle_inc = 3
        self.ridge_filter_thresh = -3

        self._mask = []
        self._normim = []
        self._orientim = []
        self._mean_freq = []
        self._median_freq = []
        self._freq = []
        self._freqim = []
        self._binim = []

    def __normalise(self, img: np.ndarray) -> np.ndarray:
        """Normalize the image.

        Args:
            img (np.ndarray): input image.

        Raises:
            ValueError: raises an exception if image is faulty.

        Returns:
            np.ndarray: normalized image
        """
        if np.std(img) == 0:
            raise ValueError("Image standard deviation is 0. Please review image again")
        normed = (img - np.mean(img)) / (np.std(img))
        return normed

    def __ridge_segment(self, img: np.ndarray):
        # RIDGESEGMENT - Normalises fingerprint image and segments ridge region
        #
        # Function identifies ridge regions of a fingerprint image and returns a
        # mask identifying this region.  It also normalises the intesity values of
        # the image so that the ridge regions have zero mean, unit standard
        # deviation.
        #
        # This function breaks the image up into blocks of size blksze x blksze and
        # evaluates the standard deviation in each region.  If the standard
        # deviation is above the threshold it is deemed part of the fingerprint.
        # Note that the image is normalised to have zero mean, unit standard
        # deviation prior to performing this process so that the threshold you
        # specify is relative to a unit standard deviation.
        #
        # Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
        #
        # Arguments:   im     - Fingerprint image to be segmented.
        #              blksze - Block size over which the the standard
        #                       deviation is determined (try a value of 16).
        #              thresh - Threshold of standard deviation to decide if a
        #                       block is a ridge region (Try a value 0.1 - 0.2)
        #
        # Ouput:     normim - Image where the ridge regions are renormalised to
        #                       have zero mean, unit standard deviation.
        #              mask   - Mask indicating ridge-like regions of the image,
        #                       0 for non ridge regions, 1 for ridge regions.
        #              maskind - Vector of indices of locations within the mask.
        #
        # Suggested values for a 500dpi fingerprint image:
        #
        #   [normim, mask, maskind] = ridgesegment(im, 16, 0.1)
        #
        # See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER

        ### REFERENCES

        # Peter Kovesi
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # pk at csse uwa edu au
        # http://www.csse.uwa.edu.au/~pk
        rows, cols = img.shape

        # normalized_im = self.__normalise(img, 0, 1)  # normalise to get zero mean and unit standard deviation
        normalized_im = self.__normalise(img)  # normalise to get zero mean and unit standard deviation

        new_rows = int(self.ridge_segment_blksze * np.ceil((float(rows)) / (float(self.ridge_segment_blksze))))
        new_cols = int(self.ridge_segment_blksze * np.ceil((float(cols)) / (float(self.ridge_segment_blksze))))

        padded_img = np.zeros((new_rows, new_cols))
        stddevim = np.zeros((new_rows, new_cols))
        padded_img[0:rows][:, 0:cols] = normalized_im
        for i in range(0, new_rows, self.ridge_segment_blksze):
            for j in range(0, new_cols, self.ridge_segment_blksze):
                block = padded_img[i : i + self.ridge_segment_blksze][:, j : j + self.ridge_segment_blksze]

                stddevim[i : i + self.ridge_segment_blksze][:, j : j + self.ridge_segment_blksze] = np.std(block) * np.ones(
                    block.shape
                )

        stddevim = stddevim[0:rows][:, 0:cols]
        self._mask = stddevim > self.ridge_segment_thresh
        mean_val = np.mean(normalized_im[self._mask])
        std_val = np.std(normalized_im[self._mask])
        self._normim = (normalized_im - mean_val) / (std_val)

    def __ridge_orient(self) -> None:
        # RIDGEORIENT - Estimates the local orientation of ridges in a fingerprint
        #
        # Usage:  [orientim, reliability, coherence] = ridgeorientation(im, gradientsigma,...
        #                                             blocksigma, ...
        #                                             orientsmoothsigma)
        #
        # Arguments:  im                - A normalised input image.
        #             gradientsigma     - Sigma of the derivative of Gaussian
        #                                 used to compute image gradients.
        #             blocksigma        - Sigma of the Gaussian weighting used to
        #                                 sum the gradient moments.
        #             orientsmoothsigma - Sigma of the Gaussian used to smooth
        #                                 the final orientation vector field.
        #                                 Optional: if ommitted it defaults to 0
        #
        # Output:    orientim          - The orientation image in radians.
        #                                 Orientation values are +ve clockwise
        #                                 and give the direction *along* the
        #                                 ridges.
        #             reliability       - Measure of the reliability of the
        #                                 orientation measure.  This is a value
        #                                 between 0 and 1. I think a value above
        #                                 about 0.5 can be considered 'reliable'.
        #                                 reliability = 1 - Imin./(Imax+.001);
        #             coherence         - A measure of the degree to which the local
        #                                 area is oriented.
        #                                 coherence = ((Imax-Imin)./(Imax+Imin)).^2;
        #
        # With a fingerprint image at a 'standard' resolution of 500dpi suggested
        # parameter values might be:
        #
        #    [orientim, reliability] = ridgeorient(im, 1, 3, 3);
        #
        # See also: RIDGESEGMENT, RIDGEFREQ, RIDGEFILTER

        ### REFERENCES

        # May 2003      Original version by Raymond Thai,
        # January 2005  Reworked by Peter Kovesi
        # October 2011  Added coherence computation and orientsmoothsigma made optional
        #
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # pk at csse uwa edu au
        # http://www.csse.uwa.edu.au/~pk

        # Calculate image gradients.
        sze = np.fix(6 * self.gradient_sigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1

        gauss = cv2.getGaussianKernel(int(sze), self.gradient_sigma)
        filter_gauss = gauss * gauss.T

        # filter_grad_x, filter_grad_y = np.gradient(filter_gauss)  # Gradient of Gaussian
        filter_grad_y, filter_grad_x = np.gradient(filter_gauss)  # Gradient of Gaussian

        gradient_x = signal.convolve2d(self._normim, filter_grad_x, mode="same")
        gradient_y = signal.convolve2d(self._normim, filter_grad_y, mode="same")

        grad_x2 = np.power(gradient_x, 2)
        grad_y2 = np.power(gradient_y, 2)
        grad_xy = gradient_x * gradient_y

        # Now smooth the covariance data to perform a weighted summation of the data.
        sze = np.fix(6 * self.block_sigma)

        gauss = cv2.getGaussianKernel(int(sze), self.block_sigma)
        filter_gauss = gauss * gauss.T

        grad_x2 = ndimage.convolve(grad_x2, filter_gauss)
        grad_y2 = ndimage.convolve(grad_y2, filter_gauss)
        grad_xy = 2 * ndimage.convolve(grad_xy, filter_gauss)

        # Analytic solution of principal direction
        denom = np.sqrt(np.power(grad_xy, 2) + np.power((grad_x2 - grad_y2), 2)) + np.finfo(float).eps

        sin_2_theta = grad_xy / denom  # Sine and cosine of doubled angles
        cos_2_theta = (grad_x2 - grad_y2) / denom

        if self.orient_smooth_sigma:
            sze = np.fix(6 * self.orient_smooth_sigma)
            if np.remainder(sze, 2) == 0:
                sze = sze + 1
            gauss = cv2.getGaussianKernel(int(sze), self.orient_smooth_sigma)
            filter_gauss = gauss * gauss.T
            cos_2_theta = ndimage.convolve(cos_2_theta, filter_gauss)  # Smoothed sine and cosine of
            sin_2_theta = ndimage.convolve(sin_2_theta, filter_gauss)  # doubled angles

        self._orientim = np.pi / 2 + np.arctan2(sin_2_theta, cos_2_theta) / 2

    def __ridge_freq(self):
        # RIDGEFREQ - Calculates a ridge frequency image
        #
        # Function to estimate the fingerprint ridge frequency across a
        # fingerprint image. This is done by considering blocks of the image and
        # determining a ridgecount within each block by a call to FREQEST.
        #
        # Usage:
        #  [freqim, medianfreq] =  ridgefreq(im, mask, orientim, blksze, windsze, ...
        #                                    minWaveLength, maxWaveLength)
        #
        # Arguments:
        #         im       - Image to be processed.
        #         mask     - Mask defining ridge regions (obtained from RIDGESEGMENT)
        #         orientim - Ridge orientation image (obtained from RIDGORIENT)
        #         blksze   - Size of image block to use (say 32)
        #         windsze  - Window length used to identify peaks. This should be
        #                    an odd integer, say 3 or 5.
        #         minWaveLength,  maxWaveLength - Minimum and maximum ridge
        #                     wavelengths, in pixels, considered acceptable.
        #
        # Output:
        #         freqim     - An image  the same size as im with  values set to
        #                      the estimated ridge spatial frequency within each
        #                      image block.  If a  ridge frequency cannot be
        #                      found within a block, or cannot be found within the
        #                      limits set by min and max Wavlength freqim is set
        #                      to zeros within that block.
        #         medianfreq - Median frequency value evaluated over all the
        #                      valid regions of the image.
        #
        # Suggested parameters for a 500dpi fingerprint image
        #   [freqim, medianfreq] = ridgefreq(im,orientim, 32, 5, 5, 15);
        #

        # See also: RIDGEORIENT, FREQEST, RIDGESEGMENT

        # Reference:
        # Hong, L., Wan, Y., and Jain, A. K. Fingerprint image enhancement:
        # Algorithm and performance evaluation. IEEE Transactions on Pattern
        # Analysis and Machine Intelligence 20, 8 (1998), 777 789.

        ### REFERENCES

        # Peter Kovesi
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # pk at csse uwa edu au
        # http://www.csse.uwa.edu.au/~pk

        rows, cols = self._normim.shape
        freq = np.zeros((rows, cols))

        for i in range(0, rows - self.ridge_freq_blksze, self.ridge_freq_blksze):
            for j in range(0, cols - self.ridge_freq_blksze, self.ridge_freq_blksze):
                blkim = self._normim[i : i + self.ridge_freq_blksze][:, j : j + self.ridge_freq_blksze]
                blkor = self._orientim[i : i + self.ridge_freq_blksze][:, j : j + self.ridge_freq_blksze]

                freq[i : i + self.ridge_freq_blksze][:, j : j + self.ridge_freq_blksze] = self.__frequest(blkim, blkor)

        self._freq = freq * self._mask
        freq_1d = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_1d[0][ind]

        self._mean_freq = np.mean(non_zero_elems_in_freq)
        self._median_freq = np.median(non_zero_elems_in_freq)  # does not work properly

        self._freq = self._mean_freq * self._mask

    def __frequest(self, blkim: np.ndarray, blkor: np.ndarray) -> np.ndarray:
        # FREQEST - Estimate fingerprint ridge frequency within image block
        #
        # Function to estimate the fingerprint ridge frequency within a small block
        # of a fingerprint image.  This function is used by RIDGEFREQ
        #
        # Usage:
        #  freqim =  freqest(im, orientim, windsze, minWaveLength, maxWaveLength)
        #
        # Arguments:
        #         im       - Image block to be processed.
        #         orientim - Ridge orientation image of image block.
        #         windsze  - Window length used to identify peaks. This should be
        #                    an odd integer, say 3 or 5.
        #         minWaveLength,  maxWaveLength - Minimum and maximum ridge
        #                     wavelengths, in pixels, considered acceptable.
        #
        # Output:
        #         freqim    - An image block the same size as im with all values
        #                     set to the estimated ridge spatial frequency.  If a
        #                     ridge frequency cannot be found, or cannot be found
        #                     within the limits set by min and max Wavlength
        #                     freqim is set to zeros.
        #
        # Suggested parameters for a 500dpi fingerprint image
        #   freqim = freqest(im,orientim, 5, 5, 15);
        #
        # See also:  RIDGEFREQ, RIDGEORIENT, RIDGESEGMENT

        ### REFERENCES

        # Peter Kovesi
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # pk at csse uwa edu au
        # http://www.csse.uwa.edu.au/~pk

        rows, _ = np.shape(blkim)

        # Find mean orientation within the block. This is done by averaging the
        # sines and cosines of the doubled angles before reconstructing the
        # angle again.  This avoids wraparound problems at the origin.

        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = math.atan2(sinorient, cosorient) / 2

        # Rotate the image block so that the ridges are vertical

        # ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/np.pi*180 + 90,1)
        # rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))
        rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode="nearest")

        # Now crop the image so that the rotated image does not contain any
        # invalid regions.  This prevents the projection down the columns
        # from being mucked up.

        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset : offset + cropsze][:, offset : offset + cropsze]

        # Sum down the columns to get a projection of the grey values down
        # the ridges.

        proj = np.sum(rotim, axis=0)
        dilation = scipy.ndimage.grey_dilation(proj, self.ridge_freq_windsze, structure=np.ones(self.ridge_freq_windsze))

        temp = np.abs(dilation - proj)

        peak_thresh = 2

        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)

        _, cols_maxind = np.shape(maxind)

        # Determine the spatial frequency of the ridges by divinding the
        # distance between the 1st and last peaks by the (No of peaks-1). If no
        # peaks are detected, or the wavelength is outside the allowed bounds,
        # the frequency image is set to 0

        if cols_maxind < 2:
            return np.zeros(blkim.shape)
        no_of_peaks = cols_maxind
        wave_length = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (no_of_peaks - 1)
        if self.min_wave_length <= wave_length <= self.max_wave_length:
            return 1 / np.double(wave_length) * np.ones(blkim.shape)
        return np.zeros(blkim.shape)

    def __ridge_filter(self):
        # RIDGEFILTER - enhances fingerprint image via oriented filters
        #
        # Function to enhance fingerprint image via oriented filters
        #
        # Usage:
        #  newim =  ridgefilter(im, orientim, freqim, kx, ky, showfilter)
        #
        # Arguments:
        #         im       - Image to be processed.
        #         orientim - Ridge orientation image, obtained from RIDGEORIENT.
        #         freqim   - Ridge frequency image, obtained from RIDGEFREQ.
        #         kx, ky   - Scale factors specifying the filter sigma relative
        #                    to the wavelength of the filter.  This is done so
        #                    that the shapes of the filters are invariant to the
        #                    scale.  kx controls the sigma in the x direction
        #                    which is along the filter, and hence controls the
        #                    bandwidth of the filter.  ky controls the sigma
        #                    across the filter and hence controls the
        #                    orientational selectivity of the filter. A value of
        #                    0.5 for both kx and ky is a good starting point.
        #         showfilter - An optional flag 0/1.  When set an image of the
        #                      largest scale filter is displayed for inspection.
        #
        # Output:
        #         newim    - The enhanced image
        #
        # See also: RIDGEORIENT, RIDGEFREQ, RIDGESEGMENT

        # Reference:
        # Hong, L., Wan, Y., and Jain, A. K. Fingerprint image enhancement:
        # Algorithm and performance evaluation. IEEE Transactions on Pattern
        # Analysis and Machine Intelligence 20, 8 (1998), 777 789.

        ### REFERENCES

        # Peter Kovesi
        # School of Computer Science & Software Engineering
        # The University of Western Australia
        # pk at csse uwa edu au
        # http://www.csse.uwa.edu.au/~pk

        norm_im = np.double(self._normim)
        rows, cols = norm_im.shape
        newim = np.zeros((rows, cols))

        freq_1d = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        # Round the array of frequencies to the nearest 0.01 to reduce the
        # number of distinct frequencies we have to deal with.

        non_zero_elems_in_freq = freq_1d[0][ind]
        non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

        unfreq = np.unique(non_zero_elems_in_freq)

        # Generate filters corresponding to these distinct frequencies and
        # orientations in 'angle_inc' increments.

        sigmax = 1 / unfreq[0] * self.relative_scale_factor_x
        sigmay = 1 / unfreq[0] * self.relative_scale_factor_y

        sze = int(np.round(3 * np.max([sigmax, sigmay])))

        mesh_x, mesh_y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

        reffilter = np.exp(-(((np.power(mesh_x, 2)) / (sigmax * sigmax) + (np.power(mesh_y, 2)) / (sigmay * sigmay)))) * np.cos(
            2 * np.pi * unfreq[0] * mesh_x
        )  # this is the original gabor filter

        filt_rows, filt_cols = reffilter.shape

        angle_range = int(180 / self.angle_inc)

        gabor_filter = np.array(np.zeros((angle_range, filt_rows, filt_cols)))

        for filter_idx in range(0, angle_range):
            # Generate rotated versions of the filter.  Note orientation
            # image provides orientation *along* the ridges, hence +90
            # degrees, and imrotate requires angles +ve anticlockwise, hence
            # the minus sign.

            rot_filt = scipy.ndimage.rotate(reffilter, -(filter_idx * self.angle_inc + 90), reshape=False)
            gabor_filter[filter_idx] = rot_filt

        # Find indices of matrix points greater than maxsze from the image
        # boundary

        maxsze = int(sze)

        temp = self._freq > 0
        validr, validc = np.where(temp)

        temp1 = validr > maxsze
        temp2 = validr < rows - maxsze
        temp3 = validc > maxsze
        temp4 = validc < cols - maxsze

        final_temp = temp1 & temp2 & temp3 & temp4

        finalind = np.where(final_temp)

        # Convert orientation matrix values from radians to an index value
        # that corresponds to round(degrees/angle_inc)

        maxorientindex = np.round(180 / self.angle_inc)
        orientindex = np.round(self._orientim / np.pi * 180 / self.angle_inc)

        # do the filtering
        for i in range(0, rows):
            for j in range(0, cols):
                if orientindex[i][j] < 1:
                    orientindex[i][j] = orientindex[i][j] + maxorientindex
                if orientindex[i][j] > maxorientindex:
                    orientindex[i][j] = orientindex[i][j] - maxorientindex
        _, finalind_cols = np.shape(finalind)
        sze = int(sze)
        for k in range(0, finalind_cols):
            cur_r = validr[finalind[0][k]]
            cur_c = validc[finalind[0][k]]

            img_block = norm_im[cur_r - sze : cur_r + sze + 1][:, cur_c - sze : cur_c + sze + 1]

            newim[cur_r][cur_c] = np.sum(img_block * gabor_filter[int(orientindex[cur_r][cur_c]) - 1])

        self._binim = newim < self.ridge_filter_thresh

    def save_enhanced_image(self, path: str) -> None:
        """Save the enhanced image to the path specified.

        Args:
            path (str): image name.
        """
        # saves the enhanced image at the specified path
        cv2.imwrite(path, (255 * self._binim))

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """Enhance the input image.

        Args:
            img (np.ndarray): input image.
            resize (bool, optional): resize the input image. Defaults to True.

        Returns:
            _type_np.ndarray: return the enhanced image.
        """
        if resize:
            rows, cols = np.shape(img)
            aspect_ratio = np.double(rows) / np.double(cols)

            new_rows = 350  # randomly selected number
            new_cols = new_rows / aspect_ratio

            img = cv2.resize(img, (int(new_cols), int(new_rows)))

        self.__ridge_segment(img)  # normalise the image and find a ROI
        self.__ridge_orient()  # compute orientation image
        self.__ridge_freq()  # compute major frequency of ridges
        self.__ridge_filter()  # filter the image using oriented gabor filter
        return self._binim
