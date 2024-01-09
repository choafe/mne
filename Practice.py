import numpy as np
import mne
from matplotlib import pyplot as plt

sample_data_folder = mne.datasets.sample.data_path()

sample_data_raw_file = (sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif" )
raw = mne.io.read_raw_fif(sample_data_raw_file)

# print(raw)
# print(raw.info)

# raw.compute_psd(fmax=50).plot(picks="data", exclude="bads")
# raw.plot(duration=5, n_channels=30)

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw, picks=ica.exclude)

orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

events = mne.find_events(raw, stim_channel="STI 014")
print(events[:5])  # show the first 5

event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "smiley": 5,
    "buttonpress": 32,
}

reject_criteria = dict(
    mag=4000e-15,  # 4000 fT
    grad=4000e-13,  # 4000 fT/cm
    eeg=150e-6,  # 150 µV
    eog=250e-6,
)  # 250 µV
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.5,
    reject=reject_criteria,
    preload=True,
)

conds_we_care_about = ["auditory/left", "auditory/right", "visual/left", "visual/right"]
epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
aud_epochs = epochs["auditory"]
vis_epochs = epochs["visual"]
del raw, epochs  # free up memory

aud_evoked = aud_epochs.average()
vis_evoked = vis_epochs.average()

aud_evoked.plot_joint(picks="eeg")
aud_evoked.plot_topomap(times=[0.0, 0.08, 0.1, 0.12, 0.2], ch_type="eeg")
plt.show(block="True")

