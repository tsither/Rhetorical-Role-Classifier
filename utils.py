from sklearn.preprocessing import LabelBinarizer

def label_encode(target_variables : list):
    """
    Encode target variables using one-hot encoding.
    
    Args:
    - target_variables (list or array-like): List of target variable strings.
    
    Returns:
    - lb (object): class object used to tranform and inverse transform.
    """
    lb = LabelBinarizer()
    lb = lb.fit(target_variables)
    
    return lb