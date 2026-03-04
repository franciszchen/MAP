### Multi-Agent In-patient Diagnostic Support (MAIDS)

#### This Code Ocean capsule provides a comprehensive environment for one-click reproducing our research results. Please note that our research is based on training and inference of Large Language Models (LLMs), which demands more GPU memory that the default Code Ocean capsule provides. As such, we use the FastAPI service  to leverage our own GPU server (IP: 127.0.0.1:6006) to present this demo of one-click reproduction. 

## 1. IPDS Benchmark 
The IPDS benchmark consists of three primary components:
*  IPDS Benchmark
    * Location: /data/Benchmark_IPDS_51274_cases.json
    * Size: 51,274 samples 
    * Purpose: Model training and development

*  Test Set
    * Location: /data/test_IPDS_1000_cases.json
    * Size: 1,000 samples
    * Usage: Comprehensive evaluation of MAIDS and baseline LLMs
    * Related Figures: 3, 4, 6, 7, 8, and 9 in the manuscript

* Clinician-Validated Test Set
    * Location: /data/test_clinician_100_cases.json
    * Size: 100 samples
    * Features: Includes assessments from three board-certified  clinical experts
    * Related Figure: 5 in the manuscript
    * Primary dataset for one-click reproduction
Note: The complete IPDS benchmark is currently under review for public access through MIMIC PhysioNet.

## 2. One-Click Reproduction for Evaluation  
To reproduce the results presented in Figure 5 of our manuscript:

~~~
bash /code/run
~~~

This script will:
1. Complete the dependency configuration
2. Initialize the FastAPI service connection
3. Execute evaluation for MAIDS

Note: The LLM temperature is set to 0.6. Due to the stochastic nature of LLM inference, slight variations in performance metrics may occur between runs. This script will take **about 1 hour** to complete the evalutation of /data/test_clinician_100_cases.json.

## 3. Pre-trained Model Access

Our trained model weights are hosted on HuggingFace (https://huggingface.co/QIOvO/maids/).

## 4. Resource Considerations

While this capsule primarily focuses on evaluation demonstrations due to Code Ocean's computational constraints, we maintain comprehensive documentation of our full implementation. For additional details, technical inquiries, or implementation concerns, please don't hesitate to reach out during the revision process.

## 5. License Information

PhysioNet Credentialed Health Data Use Agreement 1.5.0