# Quantitative Framework for Novel Image Sensor Architectures for Autonomous Edge Vision

This repository collects three related toolkits for exploring CMOS image sensor (CIS) design and performance.

The repository currently provides three modular simulators rather than a single integrated workflow. A unified interface for cross-module evaluation is a natural future extension.

## 1. CIS_Model

This is the conventional CIS sensor modeling framework. It provides a parameterized model for CIS arrays and their associated circuits, including pixel structures, readout paths, timing, power, and noise.

What it simulates:
- pixel-level and array-level sensor behavior
- frame timing and readout behavior
- power consumption of the pixel array and readout chain
- noise-related metrics such as SNR, dynamic range, and full well capacity

Use cases:
- conventional CIS design
- 3D-structured sensors
- coded-exposure sensors
- CNN-based or optically processed sensors

## 2. DVS_Noise_Model

This is a simplified event-based sensor noise simulator. It focuses on dynamic vision sensor (DVS) behavior and studies how circuit-level choices affect output noise.

What it simulates:
- noisy versus ideal DVS event traces
- event matching metrics such as TP, FP, and FN
- overall noise quality in terms of SNR metric

Use case:
- studying the impact of key circuit parameters such as threshold and refractory period on noise

## 3. Self_Powered_CIS_Simulator

This is a compact design-space simulator for self-powered CIS systems. It evaluates whether a sensor can operate sustainably under harvested power constraints.

What it simulates:
- frame period $T$ from a power-balance condition
- achievable FPS under self-powered operation
- FPS improvement compared with a baseline design

Use cases:
- exploring the tradeoff between resolution, sampling rate, and system weight
- evaluating whether a self-powered CIS can achieve a target frame rate
