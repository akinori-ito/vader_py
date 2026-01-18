import numpy as np
from sklearn.cluster import KMeans
import librosa
import pandas as pd


def voice_activity(x, simple=True, minlen=50, maxlen=1000, nclust=4, frameshift=0.01):
    """
    Voice Activity Detection function
    
    Parameters:
    -----------
    x : str or numpy.ndarray
        Path to wave file or MFCC features matrix
    simple : bool
        Use simple or heavy VAD algorithm (default: True)
    minlen : int
        Minimum duration of segment (number of frames) (default: 50)
    maxlen : int
        Maximum duration of segment (number of frames) (default: 1000)
    nclust : int
        Number of clusters, one of which should be silent (default: 4)
    frameshift : float
        The frame shift (sec), 10ms by default (default: 0.01)
    
    Returns:
    --------
    numpy.ndarray
        A boolean array with length of number of frames. 
        If the frame is a voice frame, the content is True, otherwise False.
    """
    # Check if input is a file path (string) or already MFCC features
    if isinstance(x, str):
        # Read wave file and compute MFCC
        y, sr = librosa.load(x, sr=None)
        # Compute MFCC features
        # hop_length corresponds to frameshift
        hop_length = int(frameshift * sr)
        xf = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length).T
    else:
        xf = x
    
    # Do clustering on the MFCC
    kmeans = KMeans(n_clusters=nclust, random_state=42, n_init=10)
    cls_labels = kmeans.fit_predict(xf)
    
    # Calculate average power for each cluster (using first MFCC coefficient)
    pow = np.zeros(nclust)
    for cl in range(nclust):
        pow[cl] = np.mean(xf[cls_labels == cl, 0])
    
    # The cluster with the least power should be the silence
    i_silent = np.argmin(pow)
    
    if simple:
        r = vader_simple(cls_labels, i_silent, minlen)
    else:
        r = vader_heavy(cls_labels, i_silent, minlen, maxlen)
    
    # Store frameshift as an attribute (using a custom class if needed)
    # For simplicity, we'll return a tuple (result, frameshift)
    return r, frameshift


def vader_simple(cls_labels, i_silent, minlen):
    """
    Simple Voice Activity Detection algorithm
    
    Parameters:
    -----------
    cls_labels : numpy.ndarray
        Cluster labels for each frame
    i_silent : int
        Index of the silent cluster
    minlen : int
        Minimum duration of segment (number of frames)
    
    Returns:
    --------
    numpy.ndarray
        Boolean array indicating voice activity
    """
    # Run-length encoding
    dur_lengths = []
    dur_values = []
    
    if len(cls_labels) == 0:
        return np.array([], dtype=bool)
    
    current_val = cls_labels[0]
    current_len = 1
    
    for i in range(1, len(cls_labels)):
        if cls_labels[i] == current_val:
            current_len += 1
        else:
            dur_lengths.append(current_len)
            dur_values.append(current_val)
            current_val = cls_labels[i]
            current_len = 1
    
    # Don't forget the last run
    dur_lengths.append(current_len)
    dur_values.append(current_val)
    
    nsegment = len(dur_lengths)
    
    # Fix short silent segments
    for i in range(nsegment):
        if dur_lengths[i] < minlen and dur_values[i] == i_silent:
            if i == 0:
                dur_values[i] = dur_values[i + 1]
            elif dur_values[i] != dur_values[i - 1]:
                dur_values[i] = dur_values[i - 1]
    
    # Create validity array
    seg_valid = np.zeros(len(cls_labels), dtype=bool)
    j = 0
    
    for i in range(nsegment):
        if dur_values[i] == i_silent:
            seg_valid[j:j + dur_lengths[i]] = False
        else:
            seg_valid[j:j + dur_lengths[i]] = True
        j += dur_lengths[i]
    
    return seg_valid


def vader_heavy(cls_labels, i_silent, minlen, maxlen):
    """
    Heavy Voice Activity Detection algorithm using dynamic programming
    
    Parameters:
    -----------
    cls_labels : numpy.ndarray
        Cluster labels for each frame
    i_silent : int
        Index of the silent cluster
    minlen : int
        Minimum duration of segment (number of frames)
    maxlen : int
        Maximum duration of segment (number of frames)
    
    Returns:
    --------
    numpy.ndarray
        Boolean array indicating voice activity
    """
    # Cluster decision: 1=silence, 2=non-silence (converted to 0-based indexing)
    y = (cls_labels != i_silent).astype(int) + 1
    
    n = len(y)
    
    # DP array to calculate minimum distance
    w = np.full((n, maxlen, 2), np.inf)
    b = np.zeros((n, 2), dtype=int)
    
    w[0, 0, 0] = abs(y[0] - 1)
    w[0, 0, 1] = abs(y[0] - 2)
    b[0, 0] = 0
    b[0, 1] = 0
    
    for i in range(1, n):
        if i >= minlen:
            for v in range(2):
                err = abs(y[i] - (v + 1))
                # 1-v means "the other one"
                other_v = 1 - v
                # m is the optimum length of the previous segment
                m = np.argmin(w[i - 1, minlen - 1:maxlen, other_v]) + minlen
                w[i, 0, v] = err + w[i - 1, m - 1, other_v]
                b[i, v] = m
        
        for v in range(2):
            err = abs(y[i] - (v + 1))
            for j in range(1, maxlen):
                w[i, j, v] = w[i - 1, j - 1, v] + err
    
    # Backtrace
    j1 = np.argmin(w[n - 1, :, 0])
    j2 = np.argmin(w[n - 1, :, 1])
    
    if w[n - 1, j1, 0] < w[n - 1, j2, 1]:
        j = j1
        v = 0
    else:
        j = j2
        v = 1
    
    r = np.zeros(n, dtype=int)
    i = n - 1
    
    while i >= 0:
        r[i] = v
        i -= 1
        j -= 1
        if j < 0:
            if i >= 0:
                j = b[i + 1, v] - 1
                v = 1 - v
    
    return r == 1


def voice_activity(x, simple=True, minlen=50, maxlen=1000, nclust=4, frameshift=0.01):
    """
    Voice Activity Detection function
    
    Parameters:
    -----------
    x : str or numpy.ndarray
        Path to wave file or MFCC features matrix
    simple : bool
        Use simple or heavy VAD algorithm (default: True)
    minlen : int
        Minimum duration of segment (number of frames) (default: 50)
    maxlen : int
        Maximum duration of segment (number of frames) (default: 1000)
    nclust : int
        Number of clusters, one of which should be silent (default: 4)
    frameshift : float
        The frame shift (sec), 10ms by default (default: 0.01)
    
    Returns:
    --------
    numpy.ndarray
        A boolean array with length of number of frames. 
        If the frame is a voice frame, the content is True, otherwise False.
    """
    # Check if input is a file path (string) or already MFCC features
    if isinstance(x, str):
        # Read wave file and compute MFCC
        y, sr = librosa.load(x, sr=None)
        # Compute MFCC features
        # hop_length corresponds to frameshift
        hop_length = int(frameshift * sr)
        xf = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length).T
    else:
        xf = x
    
    # Do clustering on the MFCC
    kmeans = KMeans(n_clusters=nclust, random_state=42, n_init=10)
    cls_labels = kmeans.fit_predict(xf)
    
    # Calculate average power for each cluster (using first MFCC coefficient)
    pow = np.zeros(nclust)
    for cl in range(nclust):
        pow[cl] = np.mean(xf[cls_labels == cl, 0])
    
    # The cluster with the least power should be the silence
    i_silent = np.argmin(pow)
    
    if simple:
        r = vader_simple(cls_labels, i_silent, minlen)
    else:
        r = vader_heavy(cls_labels, i_silent, minlen, maxlen)
    
    # Store frameshift as an attribute (using a custom class if needed)
    # For simplicity, we'll return a tuple (result, frameshift)
    return r, frameshift


def vader_simple(cls_labels, i_silent, minlen):
    """
    Simple Voice Activity Detection algorithm
    
    Parameters:
    -----------
    cls_labels : numpy.ndarray
        Cluster labels for each frame
    i_silent : int
        Index of the silent cluster
    minlen : int
        Minimum duration of segment (number of frames)
    
    Returns:
    --------
    numpy.ndarray
        Boolean array indicating voice activity
    """
    # Run-length encoding
    dur_lengths = []
    dur_values = []
    
    if len(cls_labels) == 0:
        return np.array([], dtype=bool)
    
    current_val = cls_labels[0]
    current_len = 1
    
    for i in range(1, len(cls_labels)):
        if cls_labels[i] == current_val:
            current_len += 1
        else:
            dur_lengths.append(current_len)
            dur_values.append(current_val)
            current_val = cls_labels[i]
            current_len = 1
    
    # Don't forget the last run
    dur_lengths.append(current_len)
    dur_values.append(current_val)
    
    nsegment = len(dur_lengths)
    
    # Fix short silent segments
    for i in range(nsegment):
        if dur_lengths[i] < minlen and dur_values[i] == i_silent:
            if i == 0:
                dur_values[i] = dur_values[i + 1]
            elif dur_values[i] != dur_values[i - 1]:
                dur_values[i] = dur_values[i - 1]
    
    # Create validity array
    seg_valid = np.zeros(len(cls_labels), dtype=bool)
    j = 0
    
    for i in range(nsegment):
        if dur_values[i] == i_silent:
            seg_valid[j:j + dur_lengths[i]] = False
        else:
            seg_valid[j:j + dur_lengths[i]] = True
        j += dur_lengths[i]
    
    return seg_valid


def vader_heavy(cls_labels, i_silent, minlen, maxlen):
    """
    Heavy Voice Activity Detection algorithm using dynamic programming
    
    Parameters:
    -----------
    cls_labels : numpy.ndarray
        Cluster labels for each frame
    i_silent : int
        Index of the silent cluster
    minlen : int
        Minimum duration of segment (number of frames)
    maxlen : int
        Maximum duration of segment (number of frames)
    
    Returns:
    --------
    numpy.ndarray
        Boolean array indicating voice activity
    """
    # Cluster decision: 1=silence, 2=non-silence (converted to 0-based indexing)
    y = (cls_labels != i_silent).astype(int) + 1
    
    n = len(y)
    
    # DP array to calculate minimum distance
    w = np.full((n, maxlen, 2), np.inf)
    b = np.zeros((n, 2), dtype=int)
    
    w[0, 0, 0] = abs(y[0] - 1)
    w[0, 0, 1] = abs(y[0] - 2)
    b[0, 0] = 0
    b[0, 1] = 0
    
    for i in range(1, n):
        if i >= minlen:
            for v in range(2):
                err = abs(y[i] - (v + 1))
                # 1-v means "the other one"
                other_v = 1 - v
                # m is the optimum length of the previous segment
                m = np.argmin(w[i - 1, minlen - 1:maxlen, other_v]) + minlen
                w[i, 0, v] = err + w[i - 1, m - 1, other_v]
                b[i, v] = m
        
        for v in range(2):
            err = abs(y[i] - (v + 1))
            for j in range(1, maxlen):
                w[i, j, v] = w[i - 1, j - 1, v] + err
    
    # Backtrace
    j1 = np.argmin(w[n - 1, :, 0])
    j2 = np.argmin(w[n - 1, :, 1])
    
    if w[n - 1, j1, 0] < w[n - 1, j2, 1]:
        j = j1
        v = 0
    else:
        j = j2
        v = 1
    
    r = np.zeros(n, dtype=int)
    i = n - 1
    
    while i >= 0:
        r[i] = v
        i -= 1
        j -= 1
        if j < 0:
            if i >= 0:
                j = b[i + 1, v] - 1
                v = 1 - v
    
    return r == 1


def voice_segment(x, unit="frame", frameshift=0.01, margin=0):
    """
    Extract voice segments from voice activity detection results
    
    Parameters:
    -----------
    x : str, numpy.ndarray, or tuple
        Path to wave file, boolean array from voice_activity(), 
        or tuple (boolean array, frameshift) from voice_activity()
    unit : str
        Unit of the result. "frame" (default) means the number of frames,
        and "time" means the duration time in seconds.
    frameshift : float
        The frame shift (in second). This parameter is used only when x is a file path.
        (default: 0.01)
    margin : int or float
        Margin to be added before and after a segment.
        If unit="frame", this should be number of frames.
        If unit="time", this should be in seconds.
        (default: 0)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'begin' and 'end' indicating segment boundaries
    
    Examples:
    ---------
    >>> # From wave file
    >>> seg = voice_segment("speech.wav", unit="time")
    >>> 
    >>> # From voice_activity result
    >>> result, fs = voice_activity("speech.wav")
    >>> seg = voice_segment((result, fs), unit="time")
    """
    # Handle different input types
    if isinstance(x, str):
        # If x is a file path, run voice_activity first
        x, frameshift = voice_activity(x, frameshift=frameshift)
    elif isinstance(x, tuple) and len(x) == 2:
        # If x is a tuple (result, frameshift) from voice_activity
        x, frameshift = x
    elif isinstance(x, np.ndarray):
        # If x is just a boolean array, use provided frameshift
        pass
    else:
        raise ValueError("x must be a file path, boolean array, or tuple from voice_activity()")
    
    # Run-length encoding
    if len(x) == 0:
        return pd.DataFrame(columns=['begin', 'end'])
    
    # Create run-length encoding
    values = []
    lengths = []
    current_val = x[0]
    current_len = 1
    
    for i in range(1, len(x)):
        if x[i] == current_val:
            current_len += 1
        else:
            values.append(current_val)
            lengths.append(current_len)
            current_val = x[i]
            current_len = 1
    
    # Don't forget the last run
    values.append(current_val)
    lengths.append(current_len)
    
    n = len(lengths)
    b = 1  # 1-based indexing to match R behavior
    f_begin = []
    f_end = []
    
    for i in range(n):
        e = b + lengths[i] - 1
        if values[i]:  # If this segment is voice (True)
            f_begin.append(b)
            f_end.append(e)
        b = e + 1
    
    # Create DataFrame
    res = pd.DataFrame({
        'begin': f_begin,
        'end': f_end
    })
    
    # Convert to time if requested
    if unit == "time":
        res['begin'] = res['begin'] * frameshift
        res['end'] = res['end'] * frameshift
    
    # Add margin
    res['begin'] = res['begin'] - margin
    res['end'] = res['end'] + margin
    
    return res

# Example usage:
if __name__ == "__main__":
    # Example with a wave file
    # result, fs = voice_activity("speech.wav")
    # print(f"Voice activity detected: {result}")
    # print(f"Frame shift: {fs}")
    
    # Example with custom parameters
    # result, fs = voice_activity("speech.wav", simple=True, minlen=30, nclust=3)
    pass