import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Dict

def standardize_nodule_df(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Function to standardize CSV files exported from CIRRUS Lung Screening for temporal analysis
    Args:
        dataset (pandas.DataFrame): input dataset
    Returns:
        pandas.DataFrame: output dataset
    """
    dataset.PatientID = dataset.PatientID.astype(int)
    dataset.LesionID = dataset.LesionID.astype(int)
    dataset.StudyDate = dataset.StudyDate.astype(int)
    dataset["NoduleID"] = [
        f"{p}_{l}" for p, l in zip(dataset.PatientID, dataset.LesionID)
    ]
    dataset["AnnotationID"] = [
        f"{n}_{s}" for n, s in zip(dataset.NoduleID, dataset.StudyDate)
    ]
    return dataset

def transform(input_image, point):
    """
    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points
    Returns
    -------
    tNumpyOrigin
    """
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )


def itk_image_to_numpy_image(input_image: sitk.Image) -> Tuple[np.array, Dict]:
    """
    Parameters
    ----------
    input_image: SimpleITK image
    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format
    """

    numpyImage = sitk.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header