# Predictive Crime Analytics: An Advanced CCTV Analytics Solution

## Abstract
An automated system is needed to detect anomalies from live camera feeds, alert law enforcement, and generate detailed reports. This enhances surveillance and law enforcement efficiency, especially during nighttime, when crimes are harder to detect manually. Our system leverages the Magnitude-Contrastive Glance-and-Focus Network (MGFN) for real-time video anomaly detection, ensuring rapid response and improved crime prevention.

## Introduction
Traditional surveillance systems require manual oversight, prone to errors, and struggle with reduced visibility at night. This project introduces an automated surveillance framework that leverages weakly-supervised learning to detect anomalies and provide actionable insights. By integrating predictive analytics, our system predicts crime hotspots, trends, and offender characteristics.

## Solutions Offered
- **Spatial Analysis:** Identify crime hotspots and location-based crime distribution.
- **Temporal Trends:** Analyze crime occurrence trends by time, day, or season.
- **Offender Profiling:** Predict criminal behavior based on demographics and history.
- **Victim Analysis:** Identify vulnerable populations and areas based on socio-economic factors.
- **Police Performance:** Compare patrolling areas with crime occurrence for performance analysis.
- **Deployment Suggestions:** Train AI models to suggest police deployment plans based on predicted crimes.

## Tech Stack
- **Programming Languages:** Python, Kotlin (for Android app development)
- **Deep Learning Frameworks:** PyTorch
- **Libraries:** NumPy, scikit-learn, tensorboardx
- **Databases:** Cassandra/MongoDB (for storing key data and inferences)

## Methodology
### Framework Overview
- **Glance Module:** Extracts long-term global context using video clip-level transformers.
- **Focus Module:** Refines local features in anomalous regions with self-attentional convolution.
- **Magnitude-Contrastive Loss:** Enhances differentiation between normal and abnormal features.
- **Top-k Features:** Optimized feature selection for anomaly classification.

### Anomaly Detection
1. Video feed processed as overlapping clips.
2. Anomalies detected using MGFN, which outputs probabilities for each frame.
3. Alerts triggered for high-probability events, sending anomaly details to police stations.

### Classification Module
- Cropped video of the anomaly analyzed for:
  - Type of anomaly (e.g., assault, robbery).
  - Instruments involved (e.g., knife, gun).
  - Severity on a scale of 1-10.

### Reporting and Predictive Analysis
- Generate detailed reports including time, location, and severity.
- Predict crime hotspots and trends for proactive prevention strategies.

## Experiments
### Datasets
- **UCF-Crime:** Evaluated using AUC (Receiver Operating Characteristic curve).
- **XD-Violence:** Evaluated using Average Precision (AP).

### Results
- **UCF-Crime:** Achieved AUC of 86.7%, surpassing previous benchmarks.
- Real-time detection and reporting: 0.2 seconds for anomaly detection, 3 seconds for report generation.

## Dependencies/Show Stoppers
- High computational power required for training.
- Large storage and data requirements.
- Optimized for 16fps inputs at fixed resolutions.

## Future Scope
- **Suspect Tracking:** Across multiple cameras.
- **Crime Prevention:** Analyze historical data for predictive insights.
- **Home Security:** Integration with alarms and automated notifications for homeowners.
- **Criminal Identification:** Automated identification using camera feeds.

## Related Work
- Previous methods like RTFM and MIL-based techniques fall short in accounting for scene variations.
- MGFN introduces Feature Amplification Mechanism (FAM) and Magnitude-Contrastive Loss (MC Loss) for improved performance.

## Acknowledgements
- **Datasets:** UCF-Crime and XD-Violence creators.
- **GPU Resources:** NVIDIA for enabling efficient model training.

## References
1. Chen, Y., Liu, Z., Zhang, B., Fok, W., Qi, X., & Wu, Y. (2021). Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
2. Sultani, W., Chen, C., & Shah, M. (2018). Real-World Anomaly Detection in Surveillance Videos. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
3. Wu, Y., & Liu, Z. (2020). XD-Violence: A Benchmark for Detecting Multiple Violent Events. Proceedings of the ACM Multimedia Conference.
