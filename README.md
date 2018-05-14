# Segson

Segson is a tool for the song segmentation in full concert audio.

## Installation

```
conda env create -f environemnt.yml
pip install --editable .
```

## Usage

```
python run.py "path/to/audio.wav"
```

## Pipeline

```
+-------+    +----------+    +----------+    +---------------+    +--------------+    +----------------+    +-----------------+
| Audio |--->| Spectrum |--->| Features |--->| Normalization |--->| Segmentation |--->| Classification |--->| Post Processing |
+-------+    +----------+    +----------+    +---------------+    +--------------+    +----------------+    +-----------------+
```

## Configuration

Several parameters in segson pipeline can be adjusted using configuration dictionary. Below is the list of accepted
parameters. For their default values, see `segson/lib/util.py` file.

#### Audio

* `audio.sampling_rate` - Sampling rate into which the original signal is resampled.
* `audio.offset` - Offset from which the audio is loaded.
* `audio.duration` - Duration of the audio. If `None` is given, then whole audio is loaded.

#### Spectrum

* `spectrum.type` - Type of spectrum computation. Supported techniques are: constant-Q transform (`"cqt"`), short-time Fourier transform (`"stft"`), and melspectrogram (`"mel"`).
* `spectrum.hop_length` - Number of frames between spectrum columns.
* `spectrum.n_bins` - Number of frequency bins in spectrum. Note that this supported only in `"cqt"` and `"mel"` spectra.

#### Feature

* `feature.flux.normalization` - Variant of spectral flux feature. Supported techniques are: no normalization (`None`) and peak normalization (`"peak"`) *Default:* `None`.

#### Normalization

* `normalization.techniques` - Normalization techniques to use. Supported values are `"outlier_removal"`, `"moving_average"`, and `"min_max"` `["outlier_removal", "moving_average", "min_max"]`.
* `normalization.moving_average.window_length` - Length of window over which moving average is computed, in seconds.

#### Segmentation

* `segmentation.llr.window_length` - Length of the whole window in which log-likelihood ratio is computed, in seconds.
* `segmentation.llr.window_overlap` - Number of seconds that whole windows overlap in log-likelihood ratio is computation.
* `segmentation.peaks.pre_max` - Number of seconds on the left side of a frame where it has to be maximum.
* `segmentation.peaks.post_max` - Number of seconds on the right side of a frame where it has to be maximum.
* `segmentation.peaks.pre_avg` - Number of seconds on the left side of a frame where frames contribute to baseline computation for thresholding.
* `segmentation.peaks.post_avg` - Number of seconds on the right side of a frame where frames contribute to baseline computation for thresholding.
* `segmentation.peaks.wait` - Delay after found peak in which no frame is identified as peak, in seconds.
* `segmentation.peaks.scale_delta` - Scale of delta parameter used in thresholding.

#### Classification

* `classification.model` - Model which is used for classification. Possible options: Silence detection (`"silence_detection"`), Rule Based (`"rule_based"`), Anomaly Detection (`"anomaly_detection"`), and their ensemble (`"ensemble"`).
* `classification.window_length` - Window length in which class threshold is computed, in seconds. If `None`, the threshold is computed from the whole audio signal.
* `classification.silence_detection.scale` - Scale parameter for computation of threshold for silence detection.
* `classification.rule_based.scale_rms` - Scale parameter for computation of threshold for root-mean-square energy feature.
* `classification.rule_based.scale_centroid` - Scale parameter for computation of threshold for spectral centroid feature.
* `classification.rule_based.scale_flux` - Scale parameter for computation of threshold for spectral flux feature.
* `classification.rule_based.scale_flatness` - Scale parameter for computation of threshold for spectral flatness feature.
* `classification.anomaly_detection.scale` - Scale parameter for computation of threshold for anomaly score.
* `classification.anomaly_detection.k` - Number of neighbors in anomaly detection classification method.

#### Postprocessing

* `postprocessing.minimum_threshold` - Minimal possible length of a song segment, in seconds.
* `postprocessing.closest_neighbor` - If a short song segment is close to a greater song segment up to this length in seconds, then these two are merged together.

## License

The code is licensed under [MIT](https://opensource.org/licenses/MIT).
