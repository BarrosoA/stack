# Neural Audio Modeling Technical Manual

A production template for capturing analog audio gear (tape machines, tube preamps, overdrive/distortion pedals, guitar amplifiers, and compressors) and deploying them as real-time VST3 / AU audio plugins using PyTorch and RTNeural (JUCE / C++).

---

## 1. System Pipeline

```
1. EXCITATION AUDIO             2. GEAR RECORDING               3. DATASET PACKAGING
   training/prepare_source_audio   DAW or Audio Interface          training/dataset.py
   [dry_corpus.wav]            --> [wet_0.0.wav ... 1.0.wav]   --> [dataset.pt]
                                                                        |
                                                                        v
6. VST3 PLUGIN IN DAW           5. C++ WEIGHT EXPORT            4. NEURAL TRAINING
   plugin_template/ (JUCE C++)     training/export_rtneural.py     training/train.py
   [Neural Audio Plugin.vst3]  <-- [model_rtneural.json]       <-- [weights_best.pt]
```

### File Map
* `training/config.json`: Master configuration for network architecture, conditioning parameters, and training settings.
* `training/prepare_source_audio.py`: Generates the 180-second dry excitation test track (sweeps, drum transients, prime multitones, slew impulses, silence).
* `training/dataset.py`: Slices aligned WAV recordings into sequence chunks, appends knob parameters, and creates 80/10/10 train/val/test splits.
* `training/model.py`: PyTorch recurrent neural network class (Gated Recurrent Unit [GRU] / Long Short-Term Memory [LSTM]) with optional dry residual addition.
* `training/loss.py`: Compound loss implementation (ESR with pre-emphasis + multi-resolution STFT + DC penalty).
* `training/train.py`: Training engine with Adam optimizer, Cosine Annealing learning rate schedule, and validation checkpointing.
* `training/evaluate.py`: Sequential inference script for rendering test audio and outputting ESR / STFT metrics.
* `training/match_levels.py`: Measures LUFS / RMS differences across drive sweeps and calculates polynomial makeup gain.
* `training/export_rtneural.py`: Reorders PyTorch GRU weights to RTNeural layout, verifies numerical precision (< 1e-6 diff), and exports JSON.
* `plugin_template/`: JUCE C++ plugin project with embedded RTNeural engine, 2x oversampling, DC blocker, and silence gating.

---

## 2. Environment Setup

Python 3.10+ is required:

```powershell
pip install -r requirements.txt
```

### Dependencies
* `torch` & `torchaudio`: Deep learning framework, recurrent layers, GPU execution, and tensor math.
* `numpy` & `scipy`: DSP operations (chirps, Butterworth bandpass/highpass filtering, SOS filtering).
* `soundfile`: Audio file read and write operations.
* `pedalboard` *(Optional: `pip install pedalboard`)*: Hosts VST3 plugins inside Python when modeling software instead of hardware.
* **C++ Compiler:** CMake (>= 3.22) and a C++20 compiler (MSVC 2022 on Windows, Xcode on macOS, or GCC/Clang on Linux).

---

## 3. Step-by-Step Implementation

### Step 0: Architecture Configuration
**File:** `training/config.json`

```json
{
  "model_name": "vintage_analog_saturator",
  "model_type": "gru",
  "sample_rate": 44100,
  "conditioning_params": ["drive"],
  "hidden_size": 48,
  "use_residual": false
}
```

#### Architecture Selection Checklist
Answer these three practical questions to immediately fill out `training/config.json`:

1. **Does your hardware have knobs?**
   * **No knobs** (fixed preamp channel, fixed tape machine speed, console strip) -> `"conditioning_params": []`
   * **1 knob** (Drive, Gain, or Saturation) -> `"conditioning_params": ["drive"]`
   * **2 knobs** (Drive and Tone) -> `"conditioning_params": ["drive", "tone"]`

2. **Is it an optical compressor with a multi-second slow release?**
   * **Yes** (Teletronix LA-2A, Tube-Tech CL 1B) -> `"model_type": "lstm"`, `"hidden_size": 48`
   * **No** (Tube preamp, tape machine, overdrive, distortion, fuzz, guitar amp, VCA compressor) -> `"model_type": "gru"`

3. **Is it heavy distortion/saturation, or a transparent clean boost?**
   * **Clean boost / subtle console line amp** (wet audio is 90%+ identical to clean input) -> `"use_residual": true`, `"hidden_size": 32`
   * **Any distortion, overdrive, tape deck, fuzz, or guitar amp** (alters EQ, clips waves, squashes dynamics) -> `"use_residual": false`, `"hidden_size": 48` (or `64` for extreme metal high-gain fuzz)

---

#### Hardware Architecture Decision Guide
Detailed engineering criteria for each configuration setting:

##### 1. Circuit Scope: Neural Recurrence vs. Analytical DSP
Determine whether the target device requires recurrent neural modeling or classical DSP algorithms:

* **Suitable for Neural Recurrence:**
  * Nonlinear saturation and waveshaping (vacuum tube clipping, diode clipping, tape magnetic saturation).
  * Dynamic circuit memory (capacitor charge/discharge, power supply sag, transformer core hysteresis).
  * Passive tone filtering (tone stacks, high-frequency tape head loss, low-frequency head resonance bump).
* **Unsuitable for Black-Box Recurrence (Requires Analytical DSP or Hybrid DDSP):**
  * Time-varying delays, pitch modulation (tape wow, motor flutter, chorus, vibrato).
  * Algorithmic reverberation and long delay reflections.
  * *Physical reason:*
    1. **State Dimension Bottleneck:** A typical analog chorus or tape wow delay spans 5 ms to 20 ms (220 to 882 samples at 44.1 kHz). A 48-unit recurrent cell maintains only 48 state variables, making it physically impossible to buffer hundreds of historical audio samples without severe loss of information.
    2. **Unsynchronized Latent LFO:** Analog modulation pedals and tape capstan drift oscillate independently. In a black-box recording of dry input vs. wet output, the neural network has no access to the internal LFO phase.
    3. **Supervised Loss Collapse:** At 1 kHz, a 0.5 ms delay error inverts waveform polarity. Supervised time-domain loss (ESR, MSE) penalizes this out-of-phase condition with up to 400% signal energy error. To minimize expected error across random phase shifts, gradient descent forces recurrent weights toward zero, muting the output or heavily low-passing the audio.
  * *Implementation Strategy:* Time-varying modulation is implemented in dedicated C++ DSP modules (such as fractional delay lines with Hermite cubic interpolation, or hybrid DDSP where a network predicts delay parameters), while the recurrent neural core models nonlinear saturation, harmonic clipping, and dynamic compression.

##### 2. Recurrent Cell Selection: GRU vs LSTM
Recurrent layers are restricted to GRU and LSTM due to real-time DAW audio thread constraints:
* **MLP (Feedforward):** Lacks internal state variables, precluding frequency-dependent filtering and dynamic component sag.
* **Transformers:** Attention complexity scales quadratically with history length, exceeding real-time audio thread CPU budgets and introducing buffer lookahead latency.
* **TCN / WaveNet:** Dilated causal convolutions require circular history buffers across blocks, adding cache overhead and consuming 5x to 10x more CPU than recurrent cells.
* **RTNeural SIMD Optimization:** Vectorized C++ runtime evaluates recurrent cells sample-by-sample with zero algorithmic lookahead latency.

Selection criteria for target hardware:
* **Gated Recurrent Unit (GRU, Recommended Default):**
  * Architecture: 3 internal gates (reset, update, new candidate state) and 1 hidden state vector.
  * Computational cost: Requires 25% fewer floating-point operations than LSTM, keeping CPU consumption below 0.5% to 0.8% per core in real-time DAW playback.
  * Physical match: Continuous state memory with uniform decay rates: vacuum tubes, clipping diodes, magnetic tape, solid-state preamps, and guitar amplifiers. Research benchmarks (Wright & Valimaki 2020) demonstrate identical perceptual fidelity and ESR to LSTM on these circuits while requiring fewer computations.
* **Long Short-Term Memory (LSTM):**
  * Architecture: 4 internal gates with independent cell state `c` and hidden state `h`.
  * Computational cost: ~25% higher CPU usage than GRU.
  * Physical match: Necessary when modeling circuits with multi-rate time constants spanning vast time intervals. For example, optical compressors (Teletronix LA-2A, Tube-Tech CL 1B) feature photoresistors with complex recovery curves: fast initial attack (< 1 ms) combined with multi-second release tails. The dedicated cell state `c` maintains long-term charge retention across thousands of samples without decaying prematurely.

##### 3. Hidden State Dimensioning
Controls the number of internal state variables (neurons) in the recurrent cell:
* **Default Dimension (48 Units):** Suitable for the majority of analog circuits. Accurately models tube preamps, tape decks, and standard overdrive pedals (Tube Screamer, Blues Driver) while drawing only ~0.8% CPU per core.
* **When to use 32 units (~3,500 parameters):** Drop to 32 only for ultra-lightweight clean boosts or console utility saturation designed to run across 60+ simultaneous DAW tracks (< 0.4% CPU per core).
* **When to use 64 units (~13,000 parameters):** Bump to 64 only for extreme square-wave fuzz (Big Muff, Fuzz Face) or multi-stage cascading metal amplifiers (Peavey 5150 lead channel). Harsh clipping turns audio into sharp square blocks with near-vertical edges; 48 units will slightly round off those sharp corners, while 64 units captures the razor-sharp fuzz bite (~1.5% CPU per core).

##### 4. Residual Connection Configuration
A residual connection is an architectural bypass defined in `model.py`:
`Output = Net(x) + Dry` (if `use_residual: true`) vs `Output = Net(x)` (if `use_residual: false`).

* **When to use `true`:** Clean boosts (EP Booster, transparent buffer stages) and console preamps operating at low drive where the wet output is 90%+ identical to the dry input. The residual connection passes the linear signal directly, allowing the network to dedicate its full parameter capacity to learning the subtle harmonic distortion delta.
* **When to use `false` (Default for all distortion and tape):** If an overdrive pedal clips audio, it alters both waveform shape and phase timing. Leaving `use_residual: true` on an overdrive pedal forcefully blends the untouched dry guitar back in at 50% volume. Because the distorted wave and clean wave are out of phase, they clash destructively—scooping out midrange, erasing bass punch, and making the distortion sound like a thin, hollow, fizzy beehive. Keep `use_residual: false` on all overdrive, distortion, fuzz, tape, and amplifier circuits.

##### 5. Continuous Conditioning Parameters
Controls continuous knob parameters (e.g. `["drive"]`, `["drive", "tone"]`) mapped to `[0.0, 1.0]`:
* **Recording Step Density:**
  * 5 steps (`0.0, 0.25, 0.5, 0.75, 1.0`) provides basic linear interpolation for monotonic controls.
  * 11 steps (`0.0, 0.1, 0.2, ... 1.0`) is recommended for circuits with non-linear control tapers (e.g. drive controls that transition sharply from clean to saturated over a narrow range).
* **Multi-Knob Conditioning:** Adding a second control (e.g. `["drive", "tone"]`) expands input dimensions to `[batch, sequence_length, 3]`. Requires recording a grid of knob combinations (e.g. 5 Drive x 5 Tone = 25 audio passes).

#### Configuration Lookup Table

| Target Hardware | `model_type` | `hidden_size` | `use_residual` | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Boost / Console Preamp** (Neve 1073, EP Booster) | `gru` | 32 | `true` | Mostly linear signal; residual pass lets the network model only the harmonic delta. |
| **Tape Machine** (Studer A800, Portastudio) | `gru` | 48 | `false` | High-frequency compression and saturation alter the full signal; dry bleed causes phase issues. |
| **Overdrive / Distortion Pedal** (Tube Screamer, RAT, DS-1) | `gru` | 48 | `false` | Diode clipping and filter shaping require full waveshaping without dry bleed. |
| **High-Gain Guitar Amp / Fuzz** (5150, Dual Rectifier, Big Muff) | `gru` | 64 | `false` | Multi-stage clipping produces complex harmonic spectra requiring higher capacity. |
| **Optical Compressor** (LA-2A, CL 1B) | `lstm` | 48 | `false` | Dual cell state required to track slow photocell release curves and fast transients simultaneously. |
| **VCA / FET Compressor** (SSL Bus Comp, 1176) | `gru` | 48 | `false` | Fast solid-state attack and release envelopes track accurately with single recurrent state. |

#### Network Topology & Layer Construction
The neural network (`training/model.py`) is structured as a two-stage sequential pipeline:

```
Input Tensor: [batch, sequence_length=2048, channels=1 + num_params]
       |
       v
Stage 1: Recurrent Core (Single-Layer GRU or LSTM, hidden_size=48, batch_first=True)
       |
       v
Stage 2: Linear Readout Head (nn.Linear(hidden_size, 1))
       |
       v
Optional Bypass: Output = Net(x) + Dry (if use_residual == true)
       |
       v
Output Tensor: [batch, sequence_length=2048, 1]
```

##### 1. Single-Layer Execution Constraint
In general machine learning, stacking multiple recurrent layers is common practice. In real-time audio plugin development, `num_layers` is strictly fixed to `1`:
* **L1 Cache Locality:** A single 48-unit GRU contains exactly 7,537 float32 parameters (~30.1 KB), fitting entirely inside high-speed L1 CPU cache.
* **DAW Real-Time Thread Constraints:** Stacking two or more layers introduces inter-layer buffering overhead. In a low-latency DAW buffer (32 to 64 samples, ~0.7 ms budget), cache misses trigger buffer underruns and audible dropouts.
* **Empirical Diminishing Returns:** Peer-reviewed benchmarks (Wright & Valimaki JAES 2020) demonstrated that stacking 2 or 3 recurrent layers produced less than 0.05% ESR improvement over a single 48-unit layer on real guitar circuits while doubling CPU consumption.

##### 2. Exclusion of Dropout
Standard regularization methods like `Dropout` or `Zoneout` must not be used:
* Recurrent audio modeling requires sample-by-sample continuity of state variables (modeling capacitor and transformer memory).
* Randomly zeroing activations during forward passes corrupts physical state continuity, injecting artificial digital transient spikes and crackles into the learned audio output. Regularization is achieved through dataset diversity and Cosine Annealing learning rate decay.

##### 3. Trainable Parameter Formula
Trainable weights scale with hidden size and parameter count:
* **Input-to-Hidden Weights:** `3 * (in_channels * hidden_size)`
* **Hidden-to-Hidden Recurrent Weights:** `3 * (hidden_size * hidden_size)`
* **Gate Biases:** `6 * hidden_size` (input bias + hidden bias)
* **Linear Readout:** `(hidden_size * 1) + 1`
* *Total for 1 knob (`in_channels = 2`, `hidden_size = 48`):* `288 + 6,912 + 288 + 49 = 7,537 parameters` (30.1 KB).

---

### Step 1: Generate Universal Test Audio
**File:** `training/prepare_source_audio.py`

Synthesizes 180 seconds of audio (`training/data/dry_corpus.wav`, 44.1 kHz, 32-bit float mono) containing five excitation signals:

```powershell
python training/prepare_source_audio.py
```

#### Excitation Signal Design: Procedural vs. Musical Audio
Virtual analog system identification requires exciting every frequency and dynamic state of the target circuit:
* **Musical Audio Limitations:** Relying solely on musical instruments (guitar riffs, vocal tracks) introduces acoustic coloration, 12-tone equal temperament pitch gaps, and genre bias. A network trained strictly on guitar DI may perform poorly when processing full-frequency synth bass or percussive drums.
* **Procedural Synthetic Corpus:** A procedural corpus systematically sweeps every frequency, crest factor, and slew rate without bias. Authentic musical audio is reserved strictly for out-of-distribution test evaluation.

#### Excitation Signal Components
* **Logarithmic Chirp Sweeps (35s, 4 levels from -30 to -1 dBFS):** Sweeps 20 Hz to 20 kHz across four discrete levels to map continuous frequency response (EQ curve) and level-dependent harmonic saturation.
* **Synthesized Drum Suite (55s of kicks, snares, cymbals, claps):** High-amplitude transient peaks (> 16 dB crest factor) with sub-millisecond attack times followed by resonant shell decay. Probes circuit transient clamping, slew-rate limits, and recovery dynamics.
* **Incommensurate Multitones (35s of non-harmonic prime sines: 53, 107, 223, 439 Hz):** Because prime frequencies share no common harmonic multiples, any new frequencies appearing in the output correspond strictly to intermodulation distortion (IMD). Probes harmonic generation in tubes, diodes, and transformers without accidental harmonic reinforcement.
* **Variable-Slew Impulses (25s with 0.1ms to 10ms rise times):** Probes circuit slew-rate limits and recovery from abrupt voltage transitions.
* **Stepped White Noise & 5s Silence (30s):** Broad spectrum coverage to establish global EQ shape and anchor the zero-input noise floor, enforcing energy dissipation and preventing idle hiss or self-oscillation.

---

### Step 2: Record Target Gear
Play `dry_corpus.wav` through hardware or software and record the wet output:

#### DAW Project Setup Checklist
Before pressing record, lock your digital audio workstation (DAW) to these exact settings:
1. **Sample Rate:** Set project sample rate and interface clock strictly to **44.1 kHz** (matching `config.json`). Do not record at 48 kHz or rely on real-time DAW resampling.
2. **Bit Depth:** 32-bit float (or 24-bit fixed).
3. **Master Chain:** Completely empty. Remove all limiters, dithering, utility gain plugins, or master EQs.
4. **Interface Input Gain:** Set interface input gain once so the loudest pass stays below -1.0 dBFS. Never touch hardware knobs or interface input gain between passes.

#### Gear Knob Configurations & File Storage
Files must be placed in `training/data/`:

* **Zero-Knob Gear (Fixed Tube Preamp, Console Channel, Fixed Tape Speed):**
  * In `config.json`: Set `"conditioning_params": []`.
  * Record a single pass: `training/data/wet.wav`.
* **1-Knob Gear (Drive / Gain / Saturation):**
  * In `config.json`: Set `"conditioning_params": ["drive"]`.
  * Record 5 to 11 passes: `wet_0.0.wav`, `wet_0.25.wav`, `wet_0.5.wav`, `wet_0.75.wav`, `wet_1.0.wav`.
* **2-Knob Gear (Drive and Tone):**
  * In `config.json`: Set `"conditioning_params": ["drive", "tone"]`.
  * Record a grid: `wet_drive0.0_tone0.0.wav`, `wet_drive0.5_tone0.5.wav`, etc.

#### Recording Rules & Gain Staging Standards

##### 1. Sample Latency Alignment via Synchronization Pip
Hardware converters and internal DSP introduce buffer delay (typically 1 to 500 samples):
* **Alignment Impact:** If target `y[n]` is delayed relative to dry `x[n]` by even 1 to 2 samples, the recurrent network burns 4 to 8 hidden units acting as a digital delay line instead of modeling analog tone.
* **Alignment Procedure:** `dry_corpus.wav` starts with 0.5 seconds of silence followed by a **single-sample alignment spike (pip) at exactly t = 0.500s**. In your DAW:
  1. Zoom in on the wet recording near `0.500s` (sample 22,050).
  2. Locate the sharp vertical spike.
  3. Cut or nudge the wet track horizontally until the wet spike aligns bit-exactly with the dry spike.

##### 2. Uniform Master Pad Calculation
Analog gear naturally compresses dynamic range (+18 dB input drive yields only ~6 dB output surge; remaining 12 dB is absorbed by physical compression). At high drive, output audio may clip above 0.0 dBFS:
* **Master Pad Calibration Steps:**
  1. Record all passes (e.g. 0.0, 0.5, 1.0) into your DAW with fixed hardware gain.
  2. Solo the loudest pass (Drive = 1.0) and inspect the peak meter on the DAW Master output.
  3. If Drive = 1.0 peaks at `+3.2 dBFS`, lower your **DAW Master Output Fader by 4.2 dB** so the peak lands safely at `-1.0 dBFS`.
  4. Bounce/export **all** passes through that exact same master fader setting.
* *Result:* Every file has 1.0 dB of digital headroom, zero clipping, and 100% of the authentic dynamic compression curve is preserved across the entire knob sweep.

##### 3. Signal Representation: 32-Bit Float [-1.0, +1.0]
Digital audio samples represent physical voltage / speaker excursion bounded between [-1.0, +1.0]:
* Standardizing inputs to 32-bit float `[-1.0, +1.0]` matches the operational range of recurrent `tanh` activation gates, keeping gradients well-conditioned and eliminating numerical scaling mismatches.

---

### Step 3: Package Audio Dataset
**File:** `training/dataset.py`

Slices continuous audio into sequence chunks, stacks constant conditioning parameters along the channel dimension, and creates an 80/10/10 train/val/test split saved to `training/data/dataset.pt`.

```powershell
python training/dataset.py
```

#### Tensor Structure
Each time step contains the audio sample followed by the static parameter values:
```
Input tensor shape: (num_chunks, sequence_length=2048, channels=1 + num_params)

Sample index:      [ audio_sample,  knob_0 (drive),  knob_1 (tone) ]
Time step t=0:     [    +0.1245,          0.50,            0.75    ]
Time step t=1:     [    +0.1832,          0.50,            0.75    ]
...
Time step t=2047:  [    -0.0411,          0.50,            0.75    ]
```

#### Slicing Physics & Hyperparameters

##### 1. Audio Sequence Length Selection
The chunk length `L` balances acoustic frequency resolution against recurrent backpropagation stability:

* **Acoustic Physics Lower Bound (Frequency Resolution):**
  * A sound wave of frequency `f` requires a specific number of samples to complete one full oscillation: `Samples per cycle = Sample_Rate / Frequency`.
  * The minimum resolvable frequency in a sliced chunk of length `L` is:
    `f_min = Sample_Rate / L`
  * *If L = 512 samples (~11.6 ms at 44.1 kHz):* `f_min = 44100 / 512 = 86.1 Hz`. A 40 Hz sub-bass wave or 41 Hz low E on a bass guitar cannot complete even half an oscillation cycle inside the chunk. The recurrent core cannot resolve sub-bass behavior.
  * *If L = 1024 samples (~23.2 ms):* `f_min = 44100 / 1024 = 43.1 Hz`. Marginally fits 40 Hz, but low-end phase tracking remains noisy.
  * *If L = 2048 samples (~46.4 ms):* `f_min = 44100 / 2048 = 21.5 Hz`. Covers the entire audible spectrum down to 21.5 Hz. Inside each chunk, a 100 Hz wave completes 4.6 full oscillation cycles, providing adequate temporal context for recurrent states to lock onto low-end resonance.
* **Recurrent Machine Learning Upper Bound (BPTT Stability):**
  * Slicing continuous audio into finite sequences prevents Backpropagation Through Time (BPTT) instability. When sequence length exceeds ~4,000 samples, recurrent gradients vanish or explode over thousands of time steps, making it difficult for the optimizer to correlate current output errors with distant past states.
  * In empirical virtual analog benchmarks (Wright & Valimaki 2020), increasing `L` from 512 to 2048 reduced modeling error (ESR) from ~5% to under 0.6%, while increasing `L` from 2048 to 4096 doubled memory consumption with only a negligible 0.02% error improvement.
* **DSP & SIMD Vector Alignment:**
  * Power-of-two dimensions (2048) align with CPU vector registers (AVX2, AVX-512) and GPU tensor cores.
  * An `L` of 2048 divides cleanly into Short-Time Fourier Transform (STFT) analysis windows (512, 1024, 2048) with zero fractional padding.

##### 2. Window Hop Size Configuration
A 50% overlap ensures that transient peaks occur at various relative phase offsets within chunks rather than being arbitrarily bisected at slice edges.

##### 3. Dataset Partitioning and Random Permutation
Randomly permuting sequence chunks (seed `42`) across both time and knob positions ensures that all excitation signal types (sweeps, drum transients, multitone primes, silence) and all knob settings are uniformly distributed across train (80%), validation (10%), and test (10%) splits.

##### 4. Storage Footprint & Asset Retention
* **Dataset File Size:** Packaging 5 knob sweeps produces a ~450 MB binary (`training/data/dataset.pt`); 11 knob sweeps produces ~990 MB. Packaging completes in 20 to 30 seconds.
* **Master WAV Retention:** Do not delete `dry_corpus.wav` or your recorded `wet_*.wav` files after packaging. They are the uncompressed analog source masters required whenever you re-slice datasets, adjust pre-emphasis, or change sample rates.

---

### Step 4: Validate Architecture & Compound Loss
**Files:** `training/model.py` and `training/loss.py`

Run self-tests to verify tensor shapes and gradient backpropagation:

```powershell
python training/model.py
python training/loss.py
```

#### Dual-Domain Loss Optimization
Standard Mean Squared Error (MSE) is ineffective for audio modeling due to three physical failure modes:

* **Microscopic Phase Shifts:** Analog circuits (tape heads, tube transformers) introduce tiny frequency-dependent phase rotations (e.g. 10 microseconds at 15 kHz). To human hearing, a 10-microsecond delay is inaudible. However, in the time domain, a half-cycle shift inverts waveform polarity, producing near-100% mathematical error. When trained solely on time-domain error, the network attenuates high frequencies to minimize this penalty, resulting in a dull, muffled model.
* **Bass Energy Masking:** Low frequencies carry high physical voltage (swinging +/-0.8), while subtle harmonic overtones sit at much lower levels (around -30 dB, swinging +/-0.01). Squared error prioritizes large voltage deviations, treating subtle harmonic coloration as negligible rounding error.
* **Silent Chunk Denominator Collapse:** Calculating ESR per sequence chunk (`sum((y - y_hat)^2) / (sum(y^2) + eps)`) causes numerical explosions on silent intervals (e.g. dividing by 1e-10 produces losses in the millions, corrupting weights to NaN).

To resolve these failure modes, the pipeline uses a **Compound Loss**:

##### 1. Batch-Wide Error-to-Signal Ratio Loss
Normalized squared difference divided by target signal energy:
`ESR = mean((y_true - y_pred)^2) / (mean(y_true^2) + epsilon)`
Normalizing squared error across the entire mini-batch rather than per chunk ensures that active audio chunks absorb silent intervals, maintaining well-conditioned gradients.

##### 2. Pre-Emphasis High-Pass Filter Weighting
Applies a first-order FIR filter (`y[n] - 0.85 * y[n-1]`) to both prediction and target prior to calculating ESR. Attenuates low bass energy, shifting gradient optimization toward the high-frequency harmonic content that defines analog clipping.

##### 3. Multi-Resolution Spectral Loss Weighting
Calculates Short-Time Fourier Transforms across three window sizes (512, 1024, 2048 samples) and computes spectral convergence plus log-magnitude distance. Measures frequency-domain harmonic match independently of minor phase shifts, preserving high-frequency extension. Clamping the magnitude floor (e.g. at -70 dBFS) prevents silent frequency bins from injecting noise.

##### 4. DC Offset Loss Penalty
Squared difference between the mean prediction and mean target:
`DC = (mean(y_pred) - mean(y_true))^2 / (mean(y_true^2) + epsilon)`
Penalizes baseline DC voltage drift from nonlinear `tanh` activations, preventing clicks and thumps during DAW processing.

##### 5. Transient Burn-in Warmup Window
Omits the first 128 samples of each chunk from loss calculation. Allows the recurrent hidden state to settle from an initial zero state without penalizing startup transients.

##### 6. Gradient Scale Parity
In audio training, raw multi-resolution STFT gradients are typically ~3.2 times larger than raw ESR gradients. Weighting STFT at 0.15 to 0.3 brings time-domain and frequency-domain gradients into approximately 1:1 balance, ensuring spectral matching does not destabilize waveform tracking.

---

### Step 5: Execute Model Training
**File:** `training/train.py`

```powershell
python training/train.py
```

#### Annotated Training Telemetry Guide
During training, each completed epoch outputs an evaluation line:

```
Epoch 12/30 | Train: 0.03120 | Val: 0.02450 (ESR: 0.01820, STFT: 0.3510, DC: 0.00005) | LR: 7.2e-4 | Time: 11.2s [*BEST*]
  [1]            [2]            [3]             [4]            [5]            [6]           [7]         [8]      [9]
```

* **[1] Epoch Index:** Current epoch out of configured total (e.g. 12 of 30).
* **[2] Train Loss:** Total weighted compound loss evaluated across all training mini-batches (`1.0*ESR + 0.15*STFT + 0.05*DC`).
* **[3] Val Loss:** Total weighted compound loss evaluated on held-out validation chunks. This single scalar dictates whether the model state is saved to `weights_best.pt`.
* **[4] Val ESR (Error-to-Signal Ratio):** Time-domain waveform error. Multiply by 100 to get percentage error: `0.01820` = `1.82% waveform error` (corresponding to a 98.18% waveform match). Governed by macro-dynamic compression, clipping envelope, and attack response.
* **[5] Val STFT:** Frequency-domain spectral distance combining spectral convergence and log-magnitude error across three FFT resolutions. Measures harmonic overtone balance and EQ profile independently of minor phase rotation. Values below `0.35` indicate tight harmonic matching within ~1 dB.
* **[6] Val DC:** Mean voltage baseline displacement. Must remain below `0.001` to guarantee click-free, pop-free DAW playback.
* **[7] LR (Learning Rate):** Current optimizer step size (`7.2e-4` = `0.00072`), actively decaying via Cosine Annealing.
* **[8] Time:** Real-world wall-clock duration to process one complete epoch.
* **[9] `[*BEST*]` Indicator:** Flags that this epoch achieved a new lowest validation loss, triggering a write to `training/models/weights_best.pt`. If multiple epochs elapse without `[*BEST*]`, the model is converging.

#### Hardware Platforms & Expected Training Runtimes
Hardware selection is automatic (`cuda` -> `mps` -> `cpu`). An expensive dedicated GPU is not required:
* **NVIDIA GPU (CUDA):** ~5 to 6 minutes for 30 epochs.
* **Apple Silicon (M1 / M2 / M3 via MPS):** ~8 to 10 minutes.
* **Standard Laptop / Desktop CPU (Intel i5/i7 or AMD Ryzen):** ~15 to 20 minutes (~35 seconds per epoch).

#### Early Epoch Convergence Verification
Monitor early epoch output to detect setup or recording errors immediately:
* **Epoch 1:** `Val ESR` typically begins around `0.20 to 0.35` (20% to 35% error).
* **By Epoch 5:** `Val ESR` must drop below `0.10` (10% error) and `Val STFT` below `0.35`.
* **Convergence Abort Criteria:** If `Val ESR` remains stuck above `0.25` at Epoch 5, stop training (`Ctrl+C`). This indicates your sample latency alignment in Step 2 was misaligned by several samples, or dry and wet files were swapped.

#### Hyperparameter Calibration Guide
Tune training parameters in `training/config.json` based on the target circuit:

* **Training Duration (`epochs`: 30 to 40):**
  * For a recurrent core of 32 to 64 units operating on ~34,000 sequence chunks (a ~17:1 ratio of training samples to parameters), network representational capacity naturally saturates between 25,000 and 40,000 gradient updates.
  * Training beyond 40 epochs yields diminishing returns and risks overfitting on synthetic test artifacts. Automated checkpointing preserves the best validation state in `training/models/weights_best.pt`.
* **Learning Rate Schedule (`lr`: 1e-3, `CosineAnnealingLR`):**
  * Begins at `1.0e-3` (0.001) for rapid acquisition of macro dynamic compression and global frequency response.
  * Anneals smoothly down to `1.0e-5` (0.00001) at the final epoch. The progressive deceleration allows the optimizer to settle into a sharp local minimum without jumping across gradient ridges.
* **Batch Size (`batch_size`: 32):**
  * Slices are processed in mini-batches of 32 sequences (~1.5 seconds of total audio per parameter update).
  * Balances gradient stability: smaller batches (8 to 16) produce noisy gradients on isolated transient bursts; larger batches (64 to 128) reduce update frequency per epoch and slow convergence.
* **Gradient Norm Clipping (`grad_clip_norm`: 1.0):**
  * Caps the maximum gradient vector norm at 1.0. Mandatory for recurrent cells to prevent Backpropagation Through Time (BPTT) gradient bursts from destabilizing hidden states.

#### Hardware-Specific Metric Acceptance Standards
Acceptable error thresholds vary fundamentally by circuit type:

| Target Hardware | Target Val ESR | Target Val STFT | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **Clean Boost / Console Preamp** (Neve 1073, EP Booster) | **< 0.005 (< 0.5%)** | **< 0.15** | Circuit is mostly linear; errors above 1% indicate latency misalignment or gain scaling mismatch. |
| **Tape Machine** (Studer A800, Portastudio) | **0.015 – 0.035 (1.5% – 3.5%)** | **< 0.25** | Models smooth magnetic flux saturation, head bump resonance, and gentle transient round-off. |
| **Overdrive / Distortion Pedal** (Tube Screamer, RAT) | **0.020 – 0.045 (2.0% – 4.5%)** | **< 0.25** | Tracks diode clipping knees, mid-boost tone filters, and dynamic compression. |
| **High-Gain Guitar Amp / Fuzz** (5150, Big Muff) | **0.035 – 0.060 (3.5% – 6.0%)** | **< 0.20** | Heavy non-linear clipping lowers time-domain correlation; accuracy is judged primarily by tight STFT match. |
| **Optical Compressor** (LA-2A, CL 1B) | **0.010 – 0.030 (1.0% – 3.0%)** | **< 0.20** | Requires LSTM to track dual time constants (fast transient clamp and multi-second photocell decay). |
| **VCA / FET Compressor** (SSL Bus Comp, 1176) | **0.015 – 0.035 (1.5% – 3.5%)** | **< 0.20** | Fast attack and release curves track accurately with standard single-layer GRU. |

#### Training Diagnostics & Troubleshooting Matrix

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| **Loss explodes to `NaN` on Epoch 1** | Recurrent gradient explosion, missing gradient clipping, or silent chunk division by zero. | Verify `grad_clip_norm: 1.0` in `config.json`; verify batch-wide energy normalization in `loss.py`; lower initial learning rate to `5e-4`. |
| **Low ESR (< 0.02) but high STFT (> 0.50)** (Muffled sound / missing bite) | Waveform envelope is matched, but the model is blinded to harmonic overtones due to bass energy masking. | Increase STFT loss weight from 0.15 to 0.30; increase pre-emphasis FIR filter coefficient from 0.85 to 0.90. |
| **Low STFT (< 0.15) but high ESR (> 0.15)** (Phase smearing / hollow tone) | Frequency balance is matched, but time-domain phase tracking is wandering, or dry residual bypass is enabled on a phase-shifting circuit. | Set `use_residual: false` in `config.json`; verify bit-exact single-sample latency alignment on recorded WAV files. |
| **Validation loss stalls at high error (> 0.10) after Epoch 5** | Model underfitting; network capacity too small for complex clipping, or recorded audio is misaligned by 1–2 samples. | Verify sample latency alignment in DAW; increase `hidden_size` from 32 to 48 or 64. |
| **Train loss drops while Val loss rises after Epoch 20** (Overfitting) | Model is memorizing synthetic test signals rather than learning physical circuit behavior. | Automated checkpointing will preserve `weights_best.pt`; reduce `hidden_size` or record more continuous knob steps. |

#### Output Files
* `training/models/weights_best.pt`: Best validation checkpoint (used for RTNeural C++ export).
* `training/models/weights_last.pt`: Last epoch resume point.
* `training/models/history.json`: Epoch-by-epoch loss log.

---

### Step 6: Audio Evaluation & Verification
**File:** `training/evaluate.py`

```powershell
python training/evaluate.py
```

#### Objective & Perceptual Validation Protocol

##### 1. Sequential Inference & Buffer State Continuity
`evaluate.py` feeds audio sequentially through `weights_best.pt`, carrying recurrent hidden states `h[t]` continuously across consecutive blocks. This exactly replicates the stateful execution of the RTNeural C++ engine inside a DAW.

##### 2. Cold-Start Hidden State Equilibrium Warm-Up
When a recurrent network begins processing from cold zero states (`h_0 = 0`), the initial gate bias can produce an instantaneous DC voltage jump on sample 0:
* Slicing in `evaluate.py` feeds a 2,048-sample resting pre-roll prior to sample 0, allowing internal recurrent states to reach equilibrium and eliminating boundary click artifacts.

##### 3. Out-of-Distribution Musical Verification
Synthetic test chirps and multitone primes confirm mathematical loss convergence. To verify real-world generalization, render full-length authentic musical stems through the model and compare them against the hardware:
* **Isolated Drums:** Evaluates transient attack clamping, snare wire sizzle, and kick drum sub-bass punch.
* **Electric Guitar DI:** Evaluates chord intermodulation, harmonic saturation bite, and sustain decay.
* **Full Stereo Mix:** Evaluates multi-frequency density, stereo phase coherence, and dynamic headroom.

##### 4. Integrated Loudness Normalization Standard ITU-R BS.1770
Increasing the drive parameter boosts physical loudness by 12 dB to 18 dB:
* **Psychoacoustic Loudness Disparity:** Human hearing perceives louder audio as inherently richer in bass and treble (equal loudness contours). Comparing un-normalized renders creates cognitive bias, making higher-drive passes sound subjectively "better" regardless of modeling accuracy.
* **Level Matching Standard:** Normalize both model predictions and hardware reference recordings to a fixed integrated loudness target (e.g. -18.0 LUFS) before critical listening tests.

##### 5. DAW Phase Inversion Null Test Analysis
1. Load the ground-truth wet recording on Track 1.
2. Load the model prediction WAV on Track 2.
3. Invert phase (polarity) on Track 2.
4. Sum the tracks to inspect the residual error signal:
   * **Acceptable Residual (Quiet Harmonic Sizzle, -30 to -40 dB down):** Indicates accurate phase alignment and dynamic envelope tracking; residual consists solely of subtle non-linear harmonic differences.
   * **Loud Comb Filtering or Hollow Sound:** Indicates sample latency misalignment (even 1–2 samples) or improper residual connection usage.
   * **Boomy Low-End Residual:** Indicates phase lag in the bass range or uncorrected DC offset drift.

#### Annotated Evaluation Output Table
`evaluate.py` processes the test corpus across recorded knob settings and prints an objective terminal telemetry table:

```
========================================================================================================
EVALUATION RESULTS (Model: weights_best.pt | Arch: GRU-48 | Residual: False)
--------------------------------------------------------------------------------------------------------
Knob Setting   ESR (%)    STFT Dist   Peak Model   Peak Target   Peak Delta   Verdict
--------------------------------------------------------------------------------------------------------
Drive = 0.00   1.14%      0.0412      -6.02 dBFS   -6.00 dBFS    -0.02 dB     EXCELLENT (Clean Match)
Drive = 0.25   1.89%      0.0520      -4.10 dBFS   -4.08 dBFS    -0.02 dB     EXCELLENT (Edge of Breakup)
Drive = 0.50   2.65%      0.0684      -2.45 dBFS   -2.40 dBFS    -0.05 dB     EXCELLENT (Harmonic Saturation)
Drive = 0.75   3.42%      0.0811      -1.20 dBFS   -1.15 dBFS    -0.05 dB     GOOD (Heavy Saturation)
Drive = 1.00   4.10%      0.0935      -0.45 dBFS   -0.40 dBFS    -0.05 dB     GOOD (Full Hard Clipping)
--------------------------------------------------------------------------------------------------------
Mean Aggregate: ESR = 2.64% | STFT = 0.0672 | Peak Delta Range: [-0.02 dB, -0.05 dB]
========================================================================================================
```

##### Evaluation Telemetry Column Metrics
* **Knob Setting:** Continuous physical parameter coordinate evaluated during the pass.
* **ESR (%):** Error-to-Signal Ratio percentage measuring time-domain waveform tracking accuracy:
  * `< 3.0%`: Perceptually indistinguishable from physical hardware on double-blind ABX tests.
  * `3.0% - 6.0%`: High fidelity. Microscopic harmonic variance in high treble, suitable for production plugins.
  * `> 8.0%`: Perceptually flawed. Audible distortion mismatch, phase flanging, or envelope tracking error.
* **STFT Dist:** Multi-resolution spectral distance quantifying tonal balance and harmonic overtones:
  * `< 0.08`: Tonal curve and harmonic spectrum accurately matched.
  * `> 0.15`: Noticeable frequency response skew (audible lack of brightness or boomy bass).
* **Peak Model vs. Peak Target (dBFS):** Highest instantaneous true-peak sample value rendered by model vs. hardware.
* **Peak Delta (dB):** Mathematical difference between model peak and target peak:
  * `Within +/- 0.2 dB`: Proper dynamic headroom and non-linear saturation knee behavior.
  * `Delta > 1.5 dB`: Model is either under-saturating (failing to clamp peaks) or over-compressing (crushing dynamics).
* **Verdict:** Automated qualitative classification based on composite ESR and STFT thresholds.

#### Systematic Architecture Adjustment Decision Loop
When evaluation telemetry or null tests reveal modeling discrepancies, apply this diagnostic decision tree to adjust hyperparameters systematically:

| Diagnostic Symptom | Root Cause | Exact File to Edit | Prescribed Remediation Action |
| :--- | :--- | :--- | :--- |
| **High ESR (> 8.0%) across all knob settings** | Sub-sample latency misalignment, inverted sync pip, or incorrect residual bypass. | `training/config.json` and DAW alignment | 1. Open DAW and verify sync pip alignment at t = 0.5s down to exact sample 22,050.<br>2. In `training/config.json`, verify `"use_residual"`: set to `false` for overdrive/fuzz/distortion/tape, set to `true` only for clean linear boosts. |
| **Low ESR (< 3.0%) but muffled highs and missing bite** | Waveform loss dominated by low-frequency energy; network under-allocates capacity to high treble. | `training/config.json` | 1. In `training/config.json`, increase spectral weight under `"loss"`: set `"stft_weight": 0.35` (up from default 0.15).<br>2. In `training/config.json`, increase pre-emphasis filter coefficient under `"loss"`: set `"pre_emphasis_coeff": 0.92` (up from default 0.85) to boost high-frequency error gradients. |
| **Rounded clipping corners and missing square-wave fuzz bite** | Insufficient recurrent state dimensionality to model hard diode clipping discontinuities. | `training/config.json` | 1. In `training/config.json`, increase hidden dimension: set `"hidden_size": 64` (up from default 48).<br>2. Re-train model for 30 epochs to provide capacity for sharp waveform edges. |
| **Accurate at min and max, but poor accuracy at middle knob settings (e.g. Drive = 0.50)** | Non-linear audio taper potentiometer curve poorly approximated by sparse 5-step grid. | `training/prepare_source_audio.py` and recording pass | 1. Re-record hardware with 11 discrete knob steps (`0.0, 0.1, 0.2, ... 1.0`).<br>2. Place denser capture points around the circuit saturation knee where resistance changes non-linearly. |
| **Validation loss diverges and rises while train loss falls after epoch 20 (Overfitting)** | Recurrent memory memorizing synthetic training sequences rather than generalized circuit transfer. | `training/config.json` | 1. Automated export defaults to `training/models/weights_best.pt` (prevents degraded late-epoch weights).<br>2. In `training/config.json`, reduce capacity: set `"hidden_size": 32` or increase training corpus duration beyond 3 minutes. |

---

### Step 7: Optional Output Level Matching
**File:** `training/match_levels.py`

```powershell
python training/match_levels.py
```

Calculates perceived loudness (LUFS / RMS) across knob steps and fits a polynomial curve to provide automatic output gain compensation in the plugin.

---

### Step 8: Export Weights to RTNeural Format
**File:** `training/export_rtneural.py`

```powershell
python training/export_rtneural.py
```

*Output:* Creates `training/models/model_rtneural.json` and automatically updates `plugin_template/Source/EmbeddedModel.h`.

#### Weight Transformation & Automated C++ Embedding
1. **Gate Reordering:** PyTorch stores GRU gates in order `[reset, update, new]` (r, z, n). RTNeural requires `[update, reset, hidden]` (z, r, h).
2. **Matrix Layout:** PyTorch weight matrices are transposed to row-major format for C++ execution.
3. **Bias Merging:** Separate input and hidden bias vectors are summed into a single bias vector per gate.
4. **Numerical Validation:** Runs a sample-by-sample forward pass comparing PyTorch output against manual C++ step arithmetic. Asserts absolute error is below 1e-6.
5. **Automated Embedding (No Manual Copy-Pasting):** `export_rtneural.py` automatically writes the formatted weight JSON directly into `plugin_template/Source/EmbeddedModel.h`, wrapping it inside the C++ raw string literal. The C++ plugin project is immediately ready for compilation without touching text files.

---

### Step 9: Compile C++ VST3 Plugin
**Directory:** `plugin_template/`

The plugin template uses **JUCE 8** and **RTNeural** to build VST3 and Standalone binaries.

#### 1. Hardware Knob Mapping in JUCE C++
In `plugin_template/Source/PluginProcessor.cpp`, the RTNeural engine processes audio sample-by-sample. Match the input array to your `conditioning_params`:

* **For 1-Knob Gear (Drive):**
  ```cpp
  // Pass instantaneous audio sample and continuous drive knob (0.0 to 1.0)
  float inputSample[2] = { dryAudioSample, *driveParameter };
  float wetAudioSample = engine.processSample(inputSample);
  ```
* **For 2-Knob Gear (Drive and Tone):**
  ```cpp
  // Pass audio sample and both knob scalars
  float inputSample[3] = { dryAudioSample, *driveParameter, *toneParameter };
  float wetAudioSample = engine.processSample(inputSample);
  ```
* **For Zero-Knob Gear (Fixed Studio Preamp / Channel):**
  ```cpp
  // Pass audio sample only
  float inputSample[1] = { dryAudioSample };
  float wetAudioSample = engine.processSample(inputSample);
  ```

#### 2. Configure Oversampling
In `plugin_template/Source/PluginProcessor.cpp`:
```cpp
// Set third parameter to true to enable 2x oversampling for high-gain clipping anti-aliasing
engine.prepare(sampleRate, samplesPerBlock, true);
```
*Note on Latency:* The native recurrent neural network runs sample-by-sample with zero algorithmic latency. When 2x oversampling is enabled, the polyphase halfband IIR filter introduces a small, fixed filter phase delay that JUCE automatically reports to the DAW for PDC (Plugin Delay Compensation).

#### 3. Compile
```powershell
cmake -B build -S plugin_template
cmake --build build --config Release
```

#### 4. Binary Output Paths
* **Windows (VST3):**
  `build/GenericNeuralAudioPlugin_artefacts/Release/VST3/Neural Audio Plugin.vst3`
* **macOS (VST3 / AU):**
  `build/GenericNeuralAudioPlugin_artefacts/Release/VST3/Neural Audio Plugin.vst3`
  `build/GenericNeuralAudioPlugin_artefacts/Release/AU/Neural Audio Plugin.component`

---

## 4. Execution Sequence Summary

```powershell
# step 0: setup env
pip install -r requirements.txt

# step 1: gen source wav
python training/prepare_source_audio.py

# step 2: record dry_corpus.wav through gear -> save training/data/wet_*.wav

# step 3: slice dataset
python training/dataset.py

# step 4: run sanity test
python training/model.py
python training/loss.py

# step 5: train model
python training/train.py

# step 6: eval render audio
python training/evaluate.py

# optional step 7: fit level match curve
python training/match_levels.py

# step 8: export rtneural json
python training/export_rtneural.py

# step 9: compile vst3
cmake -B build -S plugin_template
cmake --build build --config Release
```