"""Conversion tool from Neuroscan Curry (DAT, DAP, CEO and RS3) to FIF
"""

# Author: Edwin Dalmaijer <edwin.dalmaijer@psy.ox.ac.uk>
#
# License: BSD (3-clause)

# TODO: Apply logging throughout the entire module. Follow these rules, plz:
# http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python

import os
import datetime
import calendar

import numpy as np

from ...utils import warn, verbose
from ...channels.layout import _topo_to_sphere
from ..constants import FIFF
from ..utils import _mult_cal_one, _find_channels, _create_chs
from ..meas_info import _empty_info
from ..base import _BaseRaw, _check_update_montage
from ..utils import read_str


def read_raw_curry(input_fname, montage, eog=(), misc=(), ecg=(), emg=(),
                 data_format='auto', preload=False, verbose=None):

    """Read Curry data as raw object.

    .. Note::
        If montage is not provided, the x and y coordinates are read from the
        file header. Channels that are not assigned with keywords ``eog``,
        ``ecg``, ``emg`` and ``misc`` are assigned as eeg channels. All the eeg
        channel locations are fit to a sphere when computing the z-coordinates
        for the channels. If channels assigned as eeg channels have locations
        far away from the head (i.e. x and y coordinates don't fit to a
        sphere), all the channel locations will be distorted. If you are not
        sure that the channel locations in the header are correct, it is
        probably safer to use a (standard) montage. See
        :func:`mne.channels.read_montage`

    Parameters
    ----------
    input_fname : str
        Path to the data file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions. If None,
        xy sensor locations are read from the header (``x_coord`` and
        ``y_coord`` in ``ELECTLOC``) and fit to a sphere. See the documentation
        of :func:`mne.channels.read_montage` for more information.
    eog : list | tuple | 'auto' | 'header'
        Names of channels or list of indices that should be designated
        EOG channels. If 'header', VEOG and HEOG channels assigned in the file
        header are used. If 'auto', channel names containing 'EOG' are used.
        Defaults to empty tuple.
    misc : list | tuple
        Names of channels or list of indices that should be designated
        MISC channels. Defaults to empty tuple.
    ecg : list | tuple | 'auto'
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names containing 'ECG' are used.
        Defaults to empty tuple.
    emg : list | tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names containing 'EMG' are used.
        Defaults to empty tuple.
    data_format : 'auto' | 'int16' | 'int32'
        Defines the data format the data is read in. If 'auto', it is
        determined from the file header using ``numsamples`` field.
        Defaults to 'auto'.
    preload : bool | str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of RawCurry.
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.13
    """

    return RawCurry(input_fname, montage=montage, eog=eog, misc=misc, ecg=ecg,
                  emg=emg, data_format=data_format, preload=preload,
                  verbose=verbose)


def _get_curry_info(input_fname, eog, ecg, emg, misc, data_format):

    """Helper for reading the Curry CEO, DAP, and RS3 files."""
    
    # Get the path to the input file, without the DAT extension.
    fname = os.path.splitext(input_fname)[0]

    # Create a new dict to store all file information in.
    curry_info = dict()
    
    # READ DATA PARAMETERS FROM THE DAP FILE
    # Read the relevant parts of the DAP file, and store them in the
    # curry_info dict. In order to do so, we first open and read the entire
    # DAP file. Then, we'll go through all lines (one-by-one), and find the
    # relevant stuff in there.
    # Open and read all the lines of the DAP file.
    with open(fname+'.dap', 'r') as f:
        lines = f.readlines()

    # TODO: The following is incredibly inefficient. It's implemented like this
    # because this is the most readable to humans, and because this part takes
    # very little time anyway. However, a more elegant solution would be
    # better, as the current implementation reflects badly on the programmer.

    # Go through all lines, and parse the relevant ones.
    for i, l in enumerate(lines):
        # Check if the data format is in this line. The format should be
        # 'ASCII' or 'FLOAT'.
        if 'DataFormat' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            data_format = l[si:ei].lower()
        # Check if the session label is in this line. It can be empty.
        elif 'CurrentSessionLabel' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            session_label = l[si:ei]
        # Check if the number of channels is specified in this line.
        elif 'NumChannels' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            n_channels = int(l[si:ei])
        # Check if the number of channels is specified in this line.
        elif 'NumSamples' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            n_samples = int(l[si:ei])
        # Check if the sampling frequency is specified in this line.
        elif 'SampleFreqHz' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            sfreq = int(l[si:ei])
        # Check if the year is in this line.
        elif 'StartYear' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            YYYY = l[si:ei].zfill(4)
        # Check if the month is in this line.
        elif 'StartMonth' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            MM = l[si:ei].zfill(2)
        # Check if the day is in this line.
        elif 'StartDay' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            DD = l[si:ei].zfill(2)
        # Check if the hour is in this line.
        elif 'StartHour' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            hh = l[si:ei].zfill(2)
        # Check if the minute is in this line.
        elif 'StartMin' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            mm = l[si:ei].zfill(2)
        # Check if the second is in this line.
        elif 'StartSec' in l:
            si = l.find('=') + 2
            ei = l.find('\n')
            ss = l[si:ei].zfill(2)
    # Combine the timing variables to the starting date and time.
    date = datetime.datetime(int(YYYY), int(MM), int(DD), \
        int(hh), int(mm), int(ss))
    meas_date = calendar.timegm(date.utctimetuple())

    # The following things are always undefined in Curry (included for
    # completion, to match the Neuroscan CNT file implementation).
    patient_id = 0
    patient_name = ''
    last_name = ''
    first_name = ''
    sex = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
    hand = None
    lowpass_toggle = 1
    highpass_toggle = 1
    lowcutoff = 0
    highcutoff = sfreq / 2.0
    
    # READ CHANNEL NAMES AND NUMBERS FROM RS3
    # Create a few empty dicts and tuples that will be used to store data, or
    # to keep track of certain processes later on.
    # The read_channel dict will keep track of what type of information should
    # be extracted from lines as they are read.
    read_channel = {'nr':False, 'name':False, 'pos':False, 'status':False, \
        'misc':False}
    # The channel numbers are the numbers assigned to each channel.
    ch_nrs = []
    # The channel names are the labels assigned to channels, e.g. PO3 or EOGh.
    ch_names = []
    # Cals are 'calibration factors': The signal of each channel is multiplied
    # by these to convert the logged values into a real-world unit (e.g. muV).
    cals = []
    # The channel pos(itions) are (xyz)-coordinates that indicate their
    # location on the scalp.
    pos = []
    # The status list will contain all channel statuses (0 for bad, 1 for ok).
    status = []
    # Open and read all the lines of the RS3 file.
    with open(fname+'.rs3', 'r') as f:
        lines = f.readlines()
    # Go through all lines, and collect all relevant information.
    for i, l in enumerate(lines):
        # Check if this is the start of a sensor numbers list.
        if 'NUMBERS START_LIST' in l:
            read_channel['nr'] = True
            continue
        elif 'NUMBERS END_LIST' in l:
            read_channel['nr'] = False
        # Check if this is the start of a sensor labels list.
        elif 'LABELS START_LIST' in l:
            read_channel['name'] = True
            continue
        elif 'LABELS END_LIST' in l:
            read_channel['name'] = False
        # Check if this is the start of a sensor positions list.
        elif 'SENSORS START_LIST' in l:
            read_channel['pos'] = True
            continue
        elif 'SENSORS END_LIST' in l:
            read_channel['pos'] = False
        # Check if this is the start of an 'other' sensor labels list.
        elif 'LABELS_OTHERS START_LIST' in l:
            read_channel['misc'] = True
            continue
        elif 'LABELS_OTHERS END_LIST' in l:
            read_channel['misc'] = False
        # Check if this is the start of a sensor statuses list.
        elif 'TRANSFORM START_LIST' in l:
            read_channel['status'] = True
            continue
        elif 'TRANSFORM END_LIST' in l:
            read_channel['status'] = False

        # Check what to read from the current line (if anything).
        if read_channel['nr']:
            # Store the channel number. These are listed with three spaces in
            # front of each number, and a newline at the end. These are all
            # ignored by the int function.
            ch_nrs.append(int(l))
        elif read_channel['name']:
            # Store the channel name. These are listed without spaces, and
            # with a newline at the end of the line (which we don't include).
            ch_names.append(l[:l.find('\n')])
        elif read_channel['pos']:
            # The sensor positions are logged with one or two spaces in front
            # of tab-separated values, e.g. '' -0\t-112.2\t 38.3''
            x, y, z = l.split('\t')
            pos.append(np.array([x, y, z], dtype=float))
        elif read_channel['status']:
            # Store the channel status. This os listed with three spaces in
            # front of each number, and a newline at the end. These are all
            # ignored by the int function. (Numbers are 0 or 1, as far as I
            # understand it.)
            status.append(int(l))
        elif read_channel['misc']:
            # Store the channel name. These are listed without spaces, and
            # with a newline at the end of the line (which we don't include).
            ch_names.append(l[:l.find('\n')])
            # In addition to the sensor label, store its number (counting on
            # from the existing numbers, because the numbers listed in the
            # RS3 file count from 1; we wouldn't be able to distinguish those
            # numbers from the regular sensor numbers.)
            ch_nrs.append(len(ch_nrs))
            # Finally, we also need to store the status for this channel. This
            # is not actually recorded in the RS3 file (WHY?!), so we'll just
            # assume that it's fine.
            status.append(1)

    # Transform a few lists into NumPy arrays.
    ch_nrs = np.array(ch_nrs)
    ch_names = np.array(ch_names)
    pos = np.array(pos)
    status = np.array(status)
    # Cals are 'calibration factors': The signal of each channel is multiplied
    # by these to convert the logged values into a real-world unit (e.g. muV).
    cals = np.ones(len(ch_nrs), dtype=float) * 1e-6
    # The bads list contains all channel names that weren't recording
    # properly.
    bads = list(ch_names[status==0])
    
    # READ EVENTS FROM CEO
    # Create a few variables to store data and process information in.
    # The read_events Boolean will keep track of whether lines should be
    # parsed (as we move through them).
    read_events = False
    # The n_events variable will keep track of how many events have been
    # recorded.
    n_events = 0
    # The events list will be used to store all events in the CEO file. It
    # will be converted to a NumPy array with the same organisation as the
    # events that are returned by the mne.find_events function. Recording the
    # events in the additional info has the added advantage that they do not
    # have to be read from the stimulus channel (created below).
    events = []
    # The stim_channel is essentially an additional channel: It will have 0s
    # for each sample where no event happened, and an event number at each
    # sample where an event did happen.
    stim_channel = np.zeros(n_samples, dtype=int)
    # Open and read all the lines of the CEO file.
    with open(fname+'.ceo', 'r') as f:
        lines = f.readlines()
    # Go through all lines, and collect all relevant information.
    for i, l in enumerate(lines):
        # Check if this is the start of a sensor numbers list.
        if 'NUMBER_LIST START_LIST' in l:
            read_events = True
            continue
        elif 'NUMBER_LIST END_LIST' in l:
            read_events = False
        # Check if events should be read from the lines.
        if read_events:
            # The events are logged in a somewhat weird format, e.g.:
            # '  28554\t 0\t 9\t-1\t 28554\t 28554\t 0\t 0\t 0\t 0\t 0\n'
            # The first column is the timestamp in samples, and the third
            # column is the event ID. No clue what the other columns are.
            event = l.split('\t')
            event_time = int(event[0])
            event_id = int(event[2])
            # Store the event in the events list. The format is the same as
            # that of the array returned by the mne.find_events function.
            events.append(np.array([event_time, 0, event_id], dtype=int))
            # Set the appropriate sample in the stim_channel to the event ID.
            # Note that the index will be the sample number - 1, because
            # Python starts counting at 0.
            stim_channel[event_time-1] = event_id
            # Add one to the event counter.
            n_events += 1

    # Create a new info instance.
    info = _empty_info(sfreq)
    # Add the information on high- and low-pass filters.
    if lowpass_toggle == 1:
        info['lowpass'] = highcutoff
    if highpass_toggle == 1:
        info['highpass'] = lowcutoff
    # Store the events in the info dict.
    info['events'] = np.array(events)
    # Collect the subject information in one dictionary
    subject_info = {'hand': hand, 'id': patient_id, 'sex': sex,
                    'first_name': first_name, 'last_name': last_name}

    # If no information on EOG, EMG, and ECG channels are provided,
    # automatically select them (based on what the channel names start with).
    if eog == 'auto':
        eog = _find_channels(ch_names, 'EOG')
    if ecg == 'auto':
        ecg = _find_channels(ch_names, 'ECG')
    if emg == 'auto':
        emg = _find_channels(ch_names, 'EMG')

    # Create a new channel instance.
    chs = _create_chs(ch_names, cals, FIFF.FIFFV_COIL_EEG,
                      FIFF.FIFFV_EEG_CH, eog, ecg, emg, misc)
    # Select all the EEG channels.
    eegs = [idx for idx, ch in enumerate(chs) if
            ch['coil_type'] == FIFF.FIFFV_COIL_EEG]
    # TODO: Do we need to transform the coordinates? Is already in (x,y,z)
    # format.
    #coords = _topo_to_sphere(pos, eegs)
    coords = np.copy(pos)
    # Invert the x (0) and y (1) axes.
    coords[:,0:2] *= -1
    locs = np.zeros((len(chs), 12), dtype=float)
    locs[:len(pos), :3] = coords
    for ch, loc in zip(chs, locs):
        ch.update(loc=loc)

    # Add trigger channel that we created earlier (in the variable
    # stim_channel) to the existing channels (EEG etc.).
    chan_info = {'cal': 1.0, 'logno': len(chs) + 1, 'scanno': len(chs) + 1,
                 'range': 1.0, 'unit_mul': 0., 'ch_name': 'STI 014',
                 'unit': FIFF.FIFF_UNIT_NONE,
                 'coord_frame': FIFF.FIFFV_COORD_UNKNOWN, 'loc': np.zeros(12),
                 'coil_type': FIFF.FIFFV_COIL_NONE, 'kind': FIFF.FIFFV_STIM_CH}
    chs.append(chan_info)
    
    # Store a few of the recorded variables in the curry_info variable, for
    # later use within other functions and classes.
    curry_info.update(n_samples=n_samples, n_channels=n_channels,
        stim_channel=stim_channel, data_format=data_format)
    info.update(filename=input_fname, meas_date=np.array([meas_date, 0]),
                description=str(session_label), buffer_size_sec=10., bads=bads,
                subject_info=subject_info, chs=chs)
    # Not sure what this does... Adding None to all non-specified fields in
    # the info instance?
    info._update_redundant()
    
    return info, curry_info


class RawCurry(_BaseRaw):
    """Raw object from Neuroscan Curry DAT file.

    .. Note::
        If montage is not provided, the x and y coordinates are read from the
        file header. Channels that are not assigned with keywords ``eog``,
        ``ecg``, ``emg`` and ``misc`` are assigned as eeg channels. All the eeg
        channel locations are fit to a sphere when computing the z-coordinates
        for the channels. If channels assigned as eeg channels have locations
        far away from the head (i.e. x and y coordinates don't fit to a
        sphere), all the channel locations will be distorted. If you are not
        sure that the channel locations in the header are correct, it is
        probably safer to use a (standard) montage. See
        :func:`mne.channels.read_montage`

    Parameters
    ----------
    input_fname : str
        Path to the DAT file. Note that the associated CEO, DAP, and RS3 files
        are expected to have the same name, and to be stored in the same
        directory as the DAT file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions. If None,
        (x,y,z) sensor locations are read from the RS3 file. See the
        documentation of :func:`mne.channels.read_montage` for more info.
    eog : list | tuple
        Names of channels or list of indices that should be designated
        EOG channels. If 'auto', the channel names beginning with
        ``EOG`` are used (from the RS3 file). Defaults to empty tuple.
    misc : list | tuple
        Names of channels or list of indices that should be designated
        MISC channels. If 'auto', OTHERS channels from the RS3 file are used.
        Defaults to empty tuple.
    ecg : list | tuple
        Names of channels or list of indices that should be designated
        ECG channels. If 'auto', the channel names beginning with
        ``ECG`` are used (from the RS3 file). Defaults to empty tuple.
    emg : list | tuple
        Names of channels or list of indices that should be designated
        EMG channels. If 'auto', the channel names beginning with
        ``EMG`` are used (from the RS3 file). Defaults to empty tuple.
    data_format : 'auto' | 'ascii' | 'float32'
        Defines the data format the data is read in. If 'auto', it is
        determined from the DAP file using the ``DataFormat`` field.
        Defaults to 'auto'.
    preload : bool | str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    def __init__(self, input_fname, montage, eog=(), misc=(), ecg=(), emg=(),
        data_format='auto', preload=False, verbose=None):

        # Get the absolute path to the input file. Also check whether it
        # it actually exists, and throw an error if it doesn't.
        input_fname = os.path.abspath(input_fname)
        if not os.path.isfile(input_fname):
            raise Exception("ERROR in RawCurry.__init__: File '%s' not found." \
                % (input_fname))
        # Get recording information from the Curry CEO, DAP, and RS3 files.
        # (These should be produced together with the DAT file, and act as
        # a header to the actual data.)
        info, curry_info = _get_curry_info(input_fname, eog, ecg, emg, misc,
            data_format)
        # Calculate the last sample's index number (Python starts counting at
        # 0, so the Nth sample will have index number N-1).
        last_samps = [curry_info['n_samples'] - 1]
        # Check whether the montage makes sense.
        _check_update_montage(info, montage)
        # Initialise self, using the parent _BaseRaw class' __init__ method.
        # Python 2.X
        super(RawCurry, self).__init__(
            info, preload, filenames=[input_fname], raw_extras=[curry_info],
            last_samps=last_samps, orig_format='float',
            verbose=verbose)
#        # Python 3.X
#        super().__init__(
#            info, preload, filenames=[input_fname], raw_extras=[curry_info],
#            last_samps=last_samps, orig_format='float',
#            verbose=verbose)

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):

        """FOR INTERNAL USE. Read a segment of data from a file.

        Parameters
        ----------
        data : ndarray, shape (len(idx), stop - start + 1)
            The data array. Should be modified inplace.
        idx : ndarray | slice
            The requested channel indices.
        fi : int
            The file index that must be read from.
        start : int
            The start sample in the given file.
        stop : int
            The stop sample in the given file (inclusive).
        cals : ndarray, shape (len(idx), 1)
            Channel calibrations (already sub-indexed).
        mult : ndarray, shape (len(idx), len(info['chs']) | None
            The compensation + projection + cals matrix, if applicable.
        """
        
        # We need the number of channels that are encoded in the data file,
        # and we can get this from the curry_info.
        n_channels = self._raw_extras[0]['n_channels']
        # The stim_channel is the channel with trigger info. The same number
        # of samples, but with the event number at samples where an event was
        # logged, and 0s otherwise.
        stim_ch = self._raw_extras[0]['stim_channel']
        # The data is in 'ascii' of 'float' (=numpy.float32) format. I haven't
        # actually encountered an ASCII data file before, so only binary
        # float32 reading is implemented.
        if self._raw_extras[0]['data_format'] in ['float', 'ascii']:
            dtype = self._raw_extras[0]['data_format']
        else:
            raise Exception(\
                "ERROR in %s._read_segment_file: Unrecognised data format '%s'." \
                % (__name__, self._raw_extras[0]['data_format']))

        # NumPy array that contains the numbers of channels that need to be
        # read from the data file, plus the stimulus (=trigger) channel.
        sel = np.arange(n_channels + 1)[idx]
        # Check if the stimulus channel is in the selection.
        if n_channels in sel:
            include_stim_chan = True
        else:
            include_stim_chan = False

        # Determine the first and the last sample numbers.
        first_sample = start
        last_sample = stop
        # Calculate the number of samples to read.
        n_samples = last_sample - first_sample
        
        # Each sample takes up 4 bytes in float32 encoding. This seems
        # insignificant, until you realize that with around dozens of channels
        # and sampling rates that can go up to 1000 Hz, there will be a LOT of
        # data loaded into memory. On 32-bit systems, the amount of available
        # RAM is limited. Thus, we need to be careful not to load the entire
        # file into memory at once. For this reason, we load data in chunks
        # of 1000 samples each. Note that 1 sample here includes ALL channels!
        # Calculate the start indices of each set of data.
        chunk_size = 1000 # in samples!
        chunk_indices = xrange(0, n_samples, chunk_size)
        
        # loop through all chunks of data, loading each as we go along.
        for ci in chunk_indices:
            
            # Calculate the chunk's ending index. (It's the chunk's start
            # index plus the chunk size)
            ei = ci + chunk_size
            # If this is the final chunk index, the ending index is the last
            # sample in the range.
            if ei > n_samples:
                # The ending index is the last sample in the requested range.
                ei = n_samples
                # Correct the chunk_size variable to the current chunk's size.
                chunk_size = ei - ci
            
            # Temporarily open the data file.
            if dtype == 'float':
                with open(self._filenames[fi], 'rb') as f:
                    # Calculate what the first sample in this chunk is. It's
                    # the first sample number in the requested range, plus the
                    # current chunk's index number.
                    chunk_start = first_sample + ci
                    # Go to the first sample in this chunk (the multiplication
                    # by 4 is because each sample is 4 bytes).
                    f.seek(4*n_channels*chunk_start, os.SEEK_SET)
                    # Read the requested part of the file.
                    raw = np.fromfile(f, dtype=np.float32, \
                        count=n_channels*chunk_size)
            elif dtype == 'ascii':
                raise NotImplementedError( \
                    "ERROR in %s._read_segment_file: Reading data type 'ascii' isn't actually implemented. Sorry about that." \
                    % (__name__))
            
            # Reshape the data into separate channels.
            raw = raw.reshape((chunk_size, n_channels))
            # The raw data is in a (n_samples, n_channels) format, whereas the
            # data variable is expected to be in a (n_channels, n_samples)
            # format. Thus, we need to transpose the raw data to fit in the
            # data variable.
            raw = np.transpose(raw)

            # Only include the stimulus channel if it was part of the selected
            # channels.
            if include_stim_chan:
                # Copy the raw data into the new data variable.
                data[:-1, ci:ei] = raw[sel[:-1], :] * cals[:-1,:]
                # Select the appropriate part of the event channel.
                data[-1, ci:ei] = stim_ch[ci:ei]
            else:
                # Copy the raw data into the new data variable.
                data[:, ci:ei] = raw[sel[:], :] * cals
