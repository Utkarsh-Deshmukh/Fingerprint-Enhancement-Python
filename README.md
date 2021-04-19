# Fingerprint-Enhancement-Python

Uses oriented gabor filter bank to enhance the fingerprint image. The orientation of the gabor filters is decided by the orientation of ridges in the input image. 

## Installation and Running the tests

### method 1 - use the library
  ```
  pip install fingerprint_enhancer
  ```
  
  **Usage:**
  ```
  import fingerprint_enhancer								# Load the library
  img = cv2.imread('image_path', 0)						# read input image
  out = fingerprint_enhancer.enhance_Fingerprint(img)		# enhance the fingerprint image
  cv2.imshow('enhanced_image', out);						# display the result
  cv2.waitKey(0)											# hold the display window
  ```
  - Alternatively, the script "src/example.py" can be used to run the example for this library.

### method 2 - use the source codes
1) go into the src folder
- if on "develop" branch, run the file "example.py"
- if on "master" branch, run the file file "main_enhancement.py" 

2) The sample images are stored in the "images" folder

3) The enhanced image will be stored in the "enhanced" folder

## important note:
The Develop Branch is what is up to date. Other branches might not be up to date.


## Results
![temp](https://cloud.githubusercontent.com/assets/13918778/25770604/637b3f38-31ee-11e7-818f-1f8359c96e07.jpg)

## Theory
- We use oriented gabor filters to enhance a fingerprint image. The orientation of the gabor filters are based on the orientation of the ridges. the shape of the gabor filter is based on the frequency and wavelength of the ridges.

## License
- This project is licensed under the BSD 2 License - see the LICENSE.md file for details

## Acknowledgements
- This program is based on the paper: Hong, L., Wan, Y., and Jain, A. K. 'Fingerprint image enhancement: Algorithm and performance evaluation'. IEEE Transactions on Pattern Analysis and Machine Intelligence 20, 8 (1998), pp 777-789.

- The author would like to thank Dr. Peter Kovesi (This code is a python implementation of his work)
