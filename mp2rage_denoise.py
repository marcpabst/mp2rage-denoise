from copy import deepcopy

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats
from numpy import ma
from nilearn import plotting, masking, image


def mp2rage_robust_combination(filename_uni, filename_inv1, filename_inv2, filename_output = None, multiplying_factor=1):
    """ This is a python version of the 'RobustCombination.mat' function 
    described by O'Brien et al. and originally implemented by Jose Marques.
    
    Parameters
    ----------
    filename_uni : string
        path to the uniform T1-image (UNI)
    filename_inv1 : string 
        path to the first inversion image (INV1)
    filename_inv2 : string
        path to the second inversion image (INV2)
    filename_output : string  (optional)
        path to output image
    multiplying_factor : int  (optional)
        if it's too noisy, give it a bigger value
    """
    # define relevant functions
    mp2rage_robustfunc  =  lambda inv1, inv2, beta: (inv1.conj() * inv2 - beta) / (np.square(inv1) + np.square(inv2) + 2*beta)

    rootsquares_pos  = lambda a,b,c: (-b+np.sqrt(np.square(b) -4 *a*c))/(2*a)
    rootsquares_neg  = lambda a,b,c: (-b-np.sqrt(np.square(b) -4 *a*c))/(2*a)

    # load data
    image_uni  = nib.load(filename_uni)
    image_inv1 = nib.load(filename_inv1)
    image_inv2 = nib.load(filename_inv2)

    image_uni_fdata = image_uni.get_fdata()
    image_inv1_fdata = image_inv1.get_fdata()
    image_inv2_fdata  = image_inv2.get_fdata()

    # scale UNI image values 
    if (np.amin(image_uni_fdata) >=0) and (np.amax(image_uni_fdata >= 0.51)):
        scale = lambda x: (x - np.amax(image_uni_fdata)/2) / np.amax(image_uni_fdata)
        image_uni_fdata = scale(image_uni_fdata)
 
    # correct polarity for INV1
    image_inv1_fdata = np.sign(image_uni_fdata) * image_inv1_fdata

    # MP2RAGEimg is a phase sensitive coil combination.. some more maths has to
    # be performed to get a better INV1 estimate which here is done by assuming
    # both INV2 is closer to a real phase sensitive combination
    inv1_pos = rootsquares_pos(-image_uni_fdata, image_inv2_fdata, -np.square(image_inv2_fdata) * image_uni_fdata)
    inv1_neg = rootsquares_neg(-image_uni_fdata, image_inv2_fdata, -np.square(image_inv2_fdata) * image_uni_fdata)

    image_inv1_final_fdata = deepcopy(image_inv1_fdata)

    image_inv1_final_fdata[np.abs(image_inv1_fdata - inv1_pos) >  np.abs(image_inv1_fdata - inv1_neg)] = inv1_neg[np.abs(image_inv1_fdata - inv1_pos) >  np.abs(image_inv1_fdata - inv1_neg)]
    image_inv1_final_fdata[np.abs(image_inv1_fdata - inv1_pos) <= np.abs(image_inv1_fdata - inv1_neg)] = inv1_pos[np.abs(image_inv1_fdata - inv1_pos) <= np.abs(image_inv1_fdata - inv1_neg)]
    
    noiselevel = multiplying_factor * np.mean(np.mean(np.mean(image_inv2_fdata[1:,-10:,-10:])))

    image_output = nib.Nifti1Image(mp2rage_robustfunc(image_inv1_final_fdata, image_inv2_fdata, np.square(noiselevel)), image_inv1.affine, nib.Nifti1Header())

    if filename_output:
        nib.save(image_output, filename_output)
    else:
        return image_output


def mp2rage_masked(filename_uni, filename_inv2, filename_output=None, threshold="70%"):
    """ This is an alternativ approach to get rid of the unusual noise
    distribution in the background of mp2rage images. It was inspired by
    a suggestion on GitHub.com and works as follows:
    1. thresholding the INV2 image 
    2. creating a background mask using nilearn's 'compute_background_mask'
    3. applying that mask to clear the background from noise
    
    Parameters
    ----------
    filename_uni : string
        path to the uniform T1-image (UNI)
    filename_inv2 : string
        path to the second inversion image (INV2)
    filename_output : string  (optional)
        path to output image 
    threshold : float, string (optional)
        absolute value (float) or percentage (string, e. g. '50%')
    """
    # load data
    image_uni  = nib.load(filename_uni)
    image_inv2 = nib.load(filename_inv2)

    image_uni_fdata = image_uni.get_fdata()
    image_inv2_fdata  = image_inv2.get_fdata()
    
    # scale UNI image values 
    if (np.amin(image_uni_fdata) >=0) and (np.amax(image_uni_fdata >= 0.51)):
        scale = lambda x: (x - np.amax(image_uni_fdata)/2) / np.amax(image_uni_fdata)
        image_uni_fdata = scale(image_uni_fdata)

    image_mask = masking.compute_background_mask(image.threshold_img(image_inv2,threshold))

    image_uni_fdata[image_mask.get_data()==0] = -.5

    image_output = nib.Nifti1Image(image_uni_fdata, image_uni.affine, nib.Nifti1Header())

    if filename_output:
        nib.save(image_output, filename_output)
    else:
        return image_output
