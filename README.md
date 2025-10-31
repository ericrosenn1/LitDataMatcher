Status: Early-stage prototype. Core orchestration and worker modules functional; full API integration in progress.

**Research Question Discovery & Data Matching Pipeline**
An AI-driven framework for identifying open research questions and linking them to relevant public datasets
Overview
This repository contains a modular, continuously operating research-automation pipeline that detects unresolved scientific questions from the literature, evaluates their relevance and answerability, and matches them to available datasets that could support further investigation.
The system is composed of four [or five] worker nodes, each handling a distinct aspect of the process—and a central orchestrator that manages resources, synchronization, and scheduling. Together, they form an adaptive loop integrating literature mining, data discovery, and computational resource management.

 
 
 
Core Components
1. Literature Analysis Layer (lit_analyzer.py and modules)
This layer identifies explicit, implicit, or derivable “next-step” research questions from academic text.
Supporting Files
•	lit_analyzer.py — core extraction logic
•	sentences.csv, fix_sentences.csv, fix_sentences_csv.py — preprocessing and sentence normalization
•	train_future_classifier.py, test_lit_analyzer.py — model training and testing for future-direction/limitation classification
•	extraction/ — specialized parsers for:
o	research_questions.py
o	limitations.py
o	future_directions.py
o	__init__.py
Output: structured candidate research questions with annotations and context, passed downstream to lit_gpu_worker.py.
________________________________________
2. Literature GPU Worker (lit_gpu_worker.py)
Performs advanced semantic and contextual analysis of extracted research questions.
Key Functions
•	Clusters publications across domains (meta-analysis style)
•	Scores questions for novelty, relevance, and recurrence
•	Annotates questions with data or variables needed to pursue an answer
•	Computes an “answerability score” — how feasibly a question can be addressed with new or existing data
•	Periodically updates scores based on data robustness reported by data_worker.py
________________________________________
3. Data Discovery Layer (data_worker.py)
Continuously indexes metadata from open-access repositories and tracks dataset quality.
Long-Term Goals
•	Build a live registry of publicly available datasets
•	Quantify data robustness (sample size, reproducibility, completeness)
•	Provide structured metadata back to lit_gpu_worker.py for score recalibration
________________________________________
4. Matching and Scoring Layer (matcher.py)
Acts as the system’s integrative filter.
Combines literature-derived questions and discovered datasets to identify the most promising overlaps.
Outputs:
Top 25–100 high-value question–data pairs, ready for manual review, validation, or further modeling.
________________________________________
5. Orchestration & Resource Management (orchestrator.py)
Monitors and regulates runtime performance across all workers.
Features
•	Dynamic CPU/GPU allocation (CUDA-enabled)
•	Multi-threaded execution with adaptive rate limiting
•	Live throughput monitoring and task prioritization
•	Real-time performance tuning to prevent overloads
________________________________________
Operational Flow
The pipeline operates as a self-updating analytical engine:
1.	Extracts open research questions from scientific text.
2.	Analyzes and scores their novelty and feasibility.
3.	Indexes open datasets and identifies robust data domains.
4.	Matches top-scoring questions to relevant datasets.
5.	Continuously rebalances resources and updates scores as new data appear.
This framework provides a scalable foundation for an AI-assisted discovery engine capable of linking conceptual gaps in research to real-world datasets.

Current Capabilities
Orchestration & Control
•	Rate-limiting and adaptive task scheduling
•	Async workers with dynamic quotas
•	GPU acceleration with memory control

System Monitoring
•	Live dashboard with sparklines
•	JSONL output format
•	Process isolation and graceful shutdown

Development Roadmap
What’s Working
•	Modular orchestration system
•	Worker node integration scaffold
•	Basic data simulation and output logging
•	
What’s Missing (to reach full functionality)
1. data_worker.py — Real Repository APIs
Current: simulated fetch_summary()
Needed:
•	Integration with GEO, ClinicalTrials.gov, and other public repositories
•	Metadata extraction and normalization
2. lit_gpu_worker.py — Real Question Processing
Current: mock inputs with placeholder stats
Needed:
•	Load actual outputs from lit_analyzer
•	Cluster similar questions using embeddings
•	Compute and update importance scores
•	Annotate questions with required data variables
•	Implement feedback-based rescoring
3. matcher.py — Semantic Matching
Current: basic string-based matching
Needed:
•	Embedding-based semantic similarity
•	Variable overlap and population compatibility metrics
•	Sample size adequacy and study scope evaluation
•	Weighted composite scoring algorithm
4. lit_analyzer.py — Integration
Current: standalone component
Needed:
•	Direct pipeline connection to lit_gpu_worker
•	Automatic question-topic transformation


Implementation Plan
 

Contributing
Pull requests are welcome!
If you plan major changes, please open an issue first to discuss what you’d like to modify.
For new data sources or model integrations, please document:
•	Dependencies (requirements.txt)
•	Input/output formats
•	Resource requirements (GPU, API limits, etc.)

