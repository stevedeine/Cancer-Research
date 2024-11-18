# **PD-L1 Expression Prediction in NSCLC Patients**

## **Overview**
This project investigates the application of radiomics and machine learning techniques to predict PD-L1 expression in Non-Small Cell Lung Cancer (NSCLC) patients. PD-L1 is a critical biomarker for determining immunotherapy eligibility. By comparing full-body and lung-segmented PET/CT imaging data, this study demonstrates the superior predictive power of full-body imaging using advanced machine learning and deep learning models.

---

## **Key Features**
- **Radiomics-Based Feature Extraction**: Quantitative features such as statistical, texture, and shape metrics extracted from PET/CT scans.
- **Machine Learning Models**: Implemented Random Forest and XGBoost, optimized with hyperparameter tuning.
- **Deep Learning Models**: Developed Fully Connected Neural Networks (FCNN) with advanced regularization techniques.
- **Performance Metrics**: Evaluated models using accuracy, ROC-AUC, precision, recall, and F1-score.
- **Balanced Dataset**: Used Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance.

---

## **Dataset**
- **Source**: PET/CT scans from 308 NSCLC patients.
- **Size**: 134,590 full-body slices and 43,142 lung-segmented slices.
- **Features**: Statistical, texture, and shape-based metrics.
- **Preprocessing**:
  - Normalized image intensities.
  - Resized slices to 512×512 pixels for consistency.
  - Applied Gaussian filters to reduce noise.

---

## **Project Plan**
| **Phase**                | **Timeline**       | **Milestones**                                      |
|--------------------------|--------------------|----------------------------------------------------|
| **Data Collection**      | Week 1            | Acquire PET/CT scans and preprocess data          |
| **Feature Extraction**   | Weeks 2-3         | Extract radiomic features using PyRadiomics        |
| **Baseline Model**       | Week 4            | Train baseline models (e.g., Logistic Regression)  |
| **Model Training**       | Weeks 5-8         | Train Random Forest, XGBoost, and FCNN models      |
| **Optimization**         | Week 9-11         | Perform hyperparameter tuning                      |
| **Evaluation & Reporting** | Week 12         | Compare models and generate final reports          |

---

## **Methodology**
1. **Feature Engineering**:
   - Radiomic features extracted using PyRadiomics.
   - Feature selection performed with ElasticNet regression and Random Forest feature importance.
2. **Model Development**:
   - **Machine Learning**:
     - Random Forest and XGBoost optimized through grid and random search.
   - **Deep Learning**:
     - Fully Connected Neural Networks (FCNN) refined with dropout layers, batch normalization, and varied optimizers.
3. **Evaluation**:
   - Metrics: Accuracy, ROC-AUC, precision, recall, F1-score.
   - Cross-validation: Applied K-fold stratified validation for robustness.
4. **Tools**:
   - Python libraries: Scikit-learn, XGBoost, TensorFlow, Keras, Pandas, PyRadiomics.
   - Visualization: Matplotlib, Seaborn.

---

## **Results**
- **Best Model**: Random Forest (Full-body dataset)
  - **Accuracy**: 99.1%
  - **ROC-AUC**: 0.998
- **Deep Learning Performance**: FCNN achieved 99% accuracy with a 0.997 ROC-AUC.
- Full-body imaging consistently outperformed lung-segmented data across all metrics.

### **Visualizations**
- **Confusion Matrices**
- **Log-Loss Plots**: Demonstrate convergence and validate model performance.
- **ROC Curves**: Highlight the model’s classification ability.

---

## **Evaluation**
The models were evaluated using the following metrics:
- **Accuracy**: Proportion of correctly classified instances.
- **ROC-AUC**: Ability to distinguish between positive and negative classes.
- **Precision, Recall, F1-Score**: Detailed metrics for classification quality.
- **Confusion Matrices**: Visualized true positives, false positives, true negatives, and false negatives.

## **Conclusions**
This project demonstrates that full-body radiomic analysis offers significant advantages over lung-segmented approaches in predicting PD-L1 expression in NSCLC patients. Key findings include:

- **Superior Predictive Accuracy**: Full-body imaging achieved a 99% accuracy and a 0.998 ROC-AUC, significantly outperforming lung-segmented models.
- **Clinical Implications**: The inclusion of systemic metabolic data in radiomic analysis provides a holistic view, improving biomarker-based patient stratification for immunotherapy.
- **Scalability and Automation**: The pipeline integrates feature extraction, model training, and evaluation, offering a scalable and non-invasive alternative to traditional PD-L1 testing.

Future work can explore:
1. **Integration with Genomic Data**: Combining radiomic features with genomic information could further enhance predictive performance.
2. **Validation on Larger Datasets**: Testing the pipeline on diverse cohorts can improve generalizability.
3. **Real-World Deployment**: Adapting the framework for clinical workflows to support personalized oncology care.

The results underline the potential of radiomics and machine learning to transform cancer diagnostics and treatment planning, providing actionable insights for patient care.

---

Cross-validation was employed to ensure model robustness and prevent overfitting.

---

## **Technologies Used**
- **Programming**: Python
- **Libraries**: 
  - PyRadiomics
  - Scikit-learn
  - XGBoost
  - TensorFlow
  - Keras
  - Pandas
  - OpenCV
  - Matplotlib
  - Seaborn
- **Tools**:
  - Jupyter Notebook
  - Git for version control
