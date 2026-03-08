import numpy as np
import cv2

def mvn_type_2_lumpy(lump_width, muimg, sigimg, numimgs: int=1):
    """
    Samples a multivariate normal distribution with mean given by 
    muimg, standard deviation given by sigimg, and with a correlation 
    structure defined by the lump_width.
    
    Parameters:
    lump_width (float): Defines the correlation structure.
    muimg (ndarray): 2D array for the mean.
    sigimg (ndarray): 2D array for the standard deviation.
    numimgs (int, optional): Number of images to generate. Defaults to 1.
    
    Returns:
    ndarray: A 2D array (if numimgs==1) or a 2D matrix of shape 
             [numpixels, numimages] flattened column-major.
    """

    # Ensure inputs are numpy arrays
    muimg = np.asarray(muimg)
    sigimg = np.asarray(sigimg)
    dim = muimg.shape
    
    # Generate the convolution kernel
    x = np.arange(dim[1])
    y = np.arange(dim[0])
    X, Y = np.meshgrid(x, y)
    
    # Python uses 0-based indexing, so mapping MATLAB's (dim - 1) / 2
    center_y = (dim[0] - 1) / 2.0
    center_x = (dim[1] - 1) / 2.0
    
    kernel_width = lump_width / np.sqrt(2)
    kernel = np.exp(-(1.0 / (2 * kernel_width**2)) * ((X - center_x)**2 + (Y - center_y)**2))
    
    # Properly normalize the kernel such that the diagonal elements of A*A' are 1
    kernel = kernel / np.sqrt(np.sum(kernel**2))
    
    # Precompute the FFT of the shifted kernel (minor optimization over original)
    K = np.fft.fft2(np.fft.fftshift(kernel))
    
    if numimgs == 1:
        # Sample an iid normal with 0 mean and variance of 1
        n = np.random.randn(*dim)
        
        # Correlate the noise via a convolution
        N = np.fft.fft2(np.fft.fftshift(n))
        img = np.real(np.fft.ifftshift(np.fft.ifft2(N * K)))
        
        # Change the variance and mean
        img = (img * sigimg) + muimg
        return img
        
    else:
        # Pre-create the images matrix
        img = np.zeros((np.prod(dim), numimgs))
        
        for i in range(numimgs):
            # Sample an iid normal with 0 mean and variance of 1
            n = np.random.randn(*dim)
            
            # Correlate the noise via a convolution
            N = np.fft.fft2(np.fft.fftshift(n))
            im = np.real(np.fft.ifftshift(np.fft.ifft2(N * K)))
            
            # Change the variance and mean
            im = (im * sigimg) + muimg
            
            # Use order='F' to flatten column-major, matching MATLAB's im(:) behavior
            img[:, i] = im.flatten(order='F')
            
        return img
    

if __name__ == "__main__":
    # 1. The direct equivalent to your MATLAB one-liner:
    # img = MVNLumpy(10, zeros(128,128), ones(128,128));
    img = mvn_type_2_lumpy(5, np.zeros((64, 64)), np.ones((64, 64)))

    # 2. Scale/Normalize the float values to the 0-255 range
    # This maps the minimum value to 0 and the maximum to 255
    img_rescaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Convert to unsigned 8-bit integer
    img_uint8 = img_rescaled.astype(np.uint8)

    # 4. Save for debug purposes
    cv2.imwrite('mvn_lumpy.png', img_uint8)
