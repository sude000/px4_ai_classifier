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

## Part 2: What We Will Do Next (The Architecture)

### 3. PyTorch Data Loading (Rolling Windows)
LSTMs and CNNs do not look at a single row; they look at a *sequence* of rows to understand trends (like a sudden drop in altitude).
*   **Action:** We will build a PyTorch `Dataset` class that chops our scaled CSVs into overlapping **Rolling Windows** (e.g., taking 50 rows, which equals 0.5 seconds of flight, to predict the final row's label). We will dynamically drop the `%time` and `flight_id` columns here.

### 4. Multi-Model Architecture & Training
We will build and compare three distinct approaches:
1.  **The Baseline (XGBoost/Random Forest):** A simple, non-sequential model. Useful to see if deep learning is even necessary.
2.  **1D-CNN (Convolutional Neural Net):** Excellent at quickly finding local "spikes" or patterns in time-series data.
3.  **LSTM (Long Short-Term Memory):** The gold standard for sequences, capable of remembering long-term dependencies (e.g., a slow battery drain leading to an engine failure).
*   **Imbalance Handling:** During training, we will pass **Class Weights** to the `CrossEntropyLoss` function so the network is heavily penalized for missing the rare failures (like Elevator), forcing it to learn them despite having less data.

### 5. Streamlit Inference UI
Once the best model is saved as a `.pt` file, we will build a web interface.
*   **Action:** A user will upload a raw drone CSV. The app will run our data pipeline in the background (Scaler + Windowing), feed it to the PyTorch model, and generate an interactive graph showing exactly when and why the AI thinks the drone failed.
