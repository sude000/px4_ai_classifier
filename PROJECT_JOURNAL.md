# PX4 AI Classifier: Engineering Journal & Concepts

This document serves as a conceptual diary of the project. It explains the "Why" and "How" behind the architectural decisions made during the Data Engineering and Preprocessing phases, and outlines the upcoming Machine Learning phases.

---

## Part 1: What We Have Done (The Foundation)

### 1. The Data Engineering Phase
Our goal was to convert 47 folders of fragmented, multi-frequency ROS bag CSVs into a single, uniform dataset.

**Concept: The "Master Clock" Synchronization (`merge_asof`)**
*   **The Problem:** Sensors record at different speeds (e.g., GPS at 5Hz, IMU at 100Hz). A Neural Network requires a fixed, uniform time step ($\Delta t$) or it will suffer from "Temporal Jitter".
*   **The Solution:** We selected the IMU (`-mavros-imu-data.csv`) as the master timeline. We used Pandas' `merge_asof(direction='backward')` to snap every other sensor's readings to the nearest preceding IMU timestamp. This guaranteed perfectly uniform row intervals without looking into the future.

**Concept: Temporal Fault Triggering (The Labels)**
*   **The Problem:** A drone flight lasts several minutes, but a mechanical failure (e.g., engine out) only happens halfway through. Labeling the entire flight as a "failure" teaches the AI junk data (e.g., calling normal takeoff an "engine failure").
*   **The Solution:** We read the `failure_status-*.csv` files. We labeled rows as `0` (Normal) *until* the ground truth value became `> 0`. From that exact millisecond onward, we assigned the specific integer class (1: Engine, 2: Elevator, 3: Rudder, 4: Aileron, 5: Multi-Fault).

### 2. The Preprocessing Phase
With a single `MASTER_TRAINING_DATA.csv` created, we needed to prepare it for PyTorch.

**Concept: Feature Selection via Correlation Analysis**
*   **The Problem:** Providing redundant data doubles the computation for no gain.
*   **The Solution:** We ran a correlation matrix. We discovered that `imu_data_raw` and filtered `imu` columns had a `>0.99` correlation (they were exactly the same). We dropped the `_raw` columns to streamline the network's input. We also programmatically dropped string metadata and covariance matrices.

**Concept: Stratified Group Splitting**
*   **The Problem:** If we randomly shuffled all 12,000 rows, we would break the time-series (Row 1 goes to Train, Row 2 goes to Test). If we randomly shuffled flights, we might accidentally put all instances of a rare failure (like Elevator, which only had 2 flights) into the Test set, leaving the model blind.
*   **The Solution:** We performed a **Stratified Flight-Level Split**. We grouped the data by `flight_id` (keeping the chronological tables intact) and shuffled the *folders*, ensuring the minority classes were evenly distributed across Train (70%), Val (15%), and Test (15%).

**Concept: Scaling**
*   **The Problem:** Battery voltage is `24.0`, angular velocity is `0.001`. The network will ignore velocity simply because the number is smaller.
*   **The Solution:** We used `StandardScaler` to normalize all physical inputs to a mean of 0 and std of 1. Crucially, we fit the scaler *only* on the Training set to prevent "Data Leakage" from the validation/test flights.

---

## Part 2: PyTorch Architecture & Data Pipeline (Phase 4)

We have transitioned from raw CSV data engineering to building the actual Deep Learning pipeline. We are using a modular architecture to separate configuration, data loading, and model definitions.

### 3. Centralized Configuration (`src/config.py`)
*   **The Concept:** Hardcoding batch sizes, window lengths, and file paths across multiple scripts leads to bugs.
*   **The Implementation:** We created a single `config.py` file to hold all hyperparameters. This includes `WINDOW_SIZE = 50` (0.5 seconds of flight data) and `STEP_SIZE = 10`.

### 4. PyTorch Data Loading & Rolling Windows (`src/dataloader/dataset.py`)
*   **The Problem:** LSTMs and CNNs require 3D tensors `[batch_size, sequence_length, features]`. We must convert our 2D tabular CSV into overlapping 3D sequences without crossing flight boundaries.
*   **The Implementation:** We built `PX4FlightDataset`.
    1.  **Boundary Safety:** It groups data by `flight_id` first, ensuring a rolling window never contains the end of Flight A and the beginning of Flight B.
    2.  **Labeling the Window:** For a window of 50 rows, it assigns the `target_label` of the *very last row* (Row 50) as the label for the entire sequence.
    3.  **The DataLoader:** The `get_dataloaders()` function bundles these windows into batches of 64 and handles the shuffling, completely separating data delivery from model training.

---

## Part 3: What We Will Do Next (Models & Training)

### 5. Multi-Model Architecture (`src/models.py`)
We will build and compare three distinct approaches:
1.  **The Baseline (XGBoost):** A simple, non-sequential model. Useful to see if deep learning is even necessary.
2.  **1D-CNN (Convolutional Neural Net):** Excellent at quickly finding local "spikes" or patterns in time-series data.
3.  **LSTM (Long Short-Term Memory):** The gold standard for sequences, capable of remembering long-term context.

### 6. The Training Engine (`src/train.py`)
*   **Imbalance Handling:** During training, we will pass **Class Weights** to the `CrossEntropyLoss` function so the network is heavily penalized for missing the rare failures (like Elevator), forcing it to learn them despite having less data.
*   **Tracking:** We will integrate MLflow to track validation accuracy and loss for the different models.

### 7. Streamlit Inference UI
Once the best model is saved as a `.pt` file, we will build a web interface. Users will upload a raw drone CSV, and the app will generate an interactive graph showing exactly when and why the AI thinks the drone failed.

---

## Part 4: Real-Time Inference & Streamlit (The Delivery)

We are now moving from training to deployment. This phase turns the "black box" model into a useful engineering tool.

### 8. The Inference Engine Pillars
To make predictions on *new* data, the system follows a 4-step pipeline:

1.  **Dynamic Model Loading:** Instead of hardcoding a model, the app reads the architecture and weights dynamically. This allows us to swap a CNN for a Hybrid model without rewriting the UI code.
2.  **Live Feature Scaling:** Raw sensor data (e.g., 24V battery, 0.001 IMU drift) must be normalized. The app loads the `feature_scaler.pkl` saved during the preprocessing phase to ensure the live data "looks" exactly like the training data.
3.  **The Rolling Window Predictor:** Since the models are temporal, they require a sequence of data (e.g., the last 50 timesteps) to make one prediction. The inference engine slides a window across the uploaded CSV, generating a "Failure Probability" for every millisecond of the flight.
4.  **Temporal Highlighting:** The UI doesn't just give a single "Pass/Fail." It maps the AI's predictions back to the flight timeline, highlighting segments of the graph in different colors (Red for Engine, Yellow for Rudder, etc.).

### 9. Engineering Rule: Centralized Model Control
*   **The Concept:** The Streamlit app should never have hardcoded logic for which model version to use.
*   **The Rule:** The active model used for inference is controlled via `src/config.py`. By changing a single variable in the config, the user can switch the entire dashboard between "CNN Mode", "LSTM Mode", or "Hybrid Mode". This ensures that the training settings and the UI settings stay perfectly synchronized.
