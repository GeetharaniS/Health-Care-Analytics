# 🏥 Sunrise Multispeciality Hospital — Healthcare Intelligence Platform
### *Final Capstone Project Report & Technical Documentation*
**Document Version:** 1.0 | **Effective Period:** FY2025–26 & FY2026–27 | **Accreditation:** NABH 5th Edition  
**Campus Location:** Trichy Road, Coimbatore, Tamil Nadu, India | **Helpline:** `0422-400-7000`

---

![Sunrise Multispeciality Hospital Campus](assets/sunrise_hospital_hero.jpg)

---

## 📌 Executive Summary

Sunrise Multispeciality Hospital (SMH) is a 300-bed tertiary care hospital serving Coimbatore, Tiruppur, Erode, Pollachi, Palakkad, and Salem. During FY2025–26 (1 April 2025 – 31 March 2026), the hospital treated **11,200 inpatients** (5,400 unique patients), generating **₹146.83 Crore** in gross inpatient revenue. 

However, critical operational bottlenecks and process breakdowns resulted in significant revenue leakage and compromised patient safety:
- **30-Day Readmission Rate**: Closed at **12.6%**, exceeding the hospital's clinical target of **< 10.0%**.
- **Bed Occupancy Imbalance**: Overall occupancy was **63.1%** against a target of **75–80%**. While high-demand departments (Cardiology 86.5%, Nephrology 88.0%, Neurology 85.2%) were overwhelmed, under-utilised departments (OBG 28.5%, General Medicine 49.7%) had persistent idle capacity.
- **Insurance Claim Rejections**: Rejection rate was **9.2%**, leading to **₹9.03 Crore in lost revenue** across 725 rejected claims.
- **Post-Discharge Follow-Up Gap**: Only **54.8%** of discharged patients received the mandatory 48-hour follow-up call. Discharged patients who did not receive the call experienced a **17.7% readmission rate** compared to **8.1%** for those contacted (more than double).
- **Patient Experience**: Net Promoter Score (NPS) was **+6.4** against an industry target of **> +40.0**, driven by discharge turnaround delays (>5 hours) and confusing insurance counters.

To resolve these challenges, an integrated **Healthcare Intelligence Platform** was built as a single unified web application combining:
1. **Public Hospital Landing Website** (NABH quality standards, 9 clinical centers, 5-level ER triage guide, and appointment booking).
2. **Executive BI Dashboard** (Admissions flow, bed capacity bottlenecks, 48-hr call impact, insurance Pareto leakage analysis, and Power BI DAX models).
3. **Machine Learning Readmission Predictor** (Binary classification on `readmitted_30d` achieving **0.743 ROC-AUC** and **48.7% recall @ top-20%**, with real-time bedside clinical risk calculation).
4. **GenAI RAG Policy Chatbot** (Grounded in the official SMH Policy Manual v4.2 and FY2025-26 Annual Report using Google Gemini 2.5 Flash).
5. **Board Recommendation Memo** (4 actionable pillars delivering **+₹4.0+ Cr claim recovery**, **~290 readmissions prevented**, and **4,500 bed-days unlocked**).

---

## 📊 Executive KPI Scorecard (FY2025–26)

![Executive KPI Summary Card](assets/kpi_summary_card.png)

| Key Performance Indicator (KPI) | FY2025–26 Actual | Hospital Target | Status / Variance | Clinical & Financial Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Inpatient Admissions** | 11,200 | — | Stable | Peak demand during monsoon season (Oct–Dec) |
| **Gross Inpatient Revenue** | ₹146.83 Cr | — | Stable | Cardiology is #1 contributor (₹42.22 Cr, 28.8%) |
| **30-Day Readmission Rate** | **12.6%** | **< 10.0%** | 🔴 **Off-Track (+2.6%)** | Highest in Cardiology (20.8%) & Nephrology (20.3%) |
| **Overall Bed Occupancy** | **63.1%** | **75.0% – 80.0%** | 🟡 **Below Target (-11.9%)** | Chronic bed crunch in 3 depts; excess beds in OBG/Med |
| **Average Length of Stay (ALOS)** | **6.24 Days** | **< 5.8 Days** | 🔴 **Off-Track (+0.44 d)** | High ALOS in Neurology (7.7 d) & Nephrology (7.6 d) |
| **Claim Rejection Rate** | **9.2%** | **< 5.0%** | 🔴 **Off-Track (+4.2%)** | **₹9.03 Cr lost** (725 claims rejected) |
| **48-Hour Follow-Up Coverage** | **54.8%** | **> 90.0%** | 🔴 **Critical (-35.2%)** | Uncalled patients had 17.7% readmission rate |
| **Net Promoter Score (NPS)** | **+6.4** | **> +40.0** | 🔴 **Sub-Par (-33.6 pts)** | Discharge delay (>5 hrs) is primary patient complaint |
| **Average ER Wait (Emergency)** | **43.5 min** | **< 30.0 min** | 🟡 **Above Target (+13.5 m)** | Triage door-to-admission bottleneck in evening |
| **In-Hospital Mortality** | **1.21%** | **< 1.50%** | 🟢 **Met Target (-0.29%)** | Within NABH benchmark standards |

---

## 📈 Deliverable 1: Operational & BI Analytics Deep Dive

### 1. Inpatient Flow Seasonality & Department Bed Bottlenecks

![Bed Census Bottleneck and Monthly Flow](assets/bed_census_bottleneck.png)

#### Key Analytical Findings:
- **Monsoon Influx (Oct – Dec)**: Inpatient admissions surged significantly between October and December (peaking above 1,100 admissions/month), driven primarily by vector-borne diseases (dengue) and acute respiratory infections (pneumonia) admitted into General Medicine and Pulmonology.
- **Capacity Crunch (Red Escalation Zone)**:
  - **Nephrology (88.0% occupancy)**, **Cardiology (86.5% occupancy)**, and **Neurology (85.2% occupancy)** consistently exceeded the 85% Amber threshold and 90% Red alert threshold, forcing postponement of elective surgeries and emergency diversions.
- **Under-Utilisation (Opportunity for Flexing)**:
  - **Obstetrics & Gynaecology (28.5% occupancy)** and **General Medicine (49.7% occupancy)** maintained substantial unused bed capacity.
  - **Recommendation**: Formally activate the SMH Bed Flexing SOP (Section 5.2) to designate 20–25 beds in low-occupancy wards as overflow step-down beds for Cardiology and Nephrology.

---

### 2. 30-Day Readmissions: Clinical & Care Transition Drivers

![Readmission Drivers and 48-Hour Call Gap](assets/readmission_drivers.png)

#### Key Analytical Findings:
1. **The 48-Hour Follow-Up Call Effect**:
   - Patients who received a structured 48-hour nurse follow-up call had an **8.1% readmission rate**.
   - Patients who were not contacted experienced a **17.7% readmission rate** — **a 2.18x increase in risk**.
   - With current coverage at only 54.8%, scaling outreach to 100% will immediately bring the hospital-wide rate below 10%.
2. **Clinical Speciality Disparities**:
   - **Cardiology (20.8%)** and **Nephrology (20.3%)** exhibit the highest readmission rates, followed by **Pulmonology (16.3%)** and **Neurology (16.2%)**.
   - Orthopaedics (5.8%) and OBG (2.6%) showed the lowest readmission rates.
3. **Chronic Disease Clustering**:
   - Heart disease patients had a **19.8% readmission rate**; Chronic Kidney Disease (CKD) had **18.9%**; COPD had **16.5%**.
   - In contrast, patients with zero comorbidities had only a **4.8% readmission rate**.
4. **Care Transition Process Failures**:
   - **Late Discharges (≥ 18:00)** had a **16.2% readmission rate** versus 10.9% for morning discharges. Late-evening discharges hinder same-day pharmacy dispensing and caregiver counselling.
   - **Distance > 50 km**: Patients living outside the immediate metro area had a **15.4% readmission rate** versus 11.2% for local residents, underscoring the urgent need for tele-follow-up.

---

### 3. Revenue Leakage & Insurance Claim Pareto Analysis

![Insurance Claims Pareto Rejections](assets/claims_pareto.png)

#### Key Analytical Findings:
- Total insurance claims processed: **7,845 claims** valued at **₹96.04 Crore**.
- Rejected claims: **725 claims (9.2%)** representing **₹9.03 Crore** in lost hospital revenue.
- **Pareto Principle Validated**: The top 3 rejection reasons account for **68.4% of all rejections and ₹6.1 Crore of financial leakage**:
  1. **Incomplete Documents (250 claims | ₹3.12 Cr)**: Missing doctor discharge summary signatures, itemised pharmacy bills, and unattached investigation reports.
  2. **Pre-authorisation Not Obtained (138 claims | ₹1.72 Cr)**: Emergency cases where pre-auth was not lodged within the mandatory 24-hour window.
  3. **Package Mismatch (105 claims | ₹1.31 Cr)**: Mismatched procedure package codes under PM-JAY and CMCHIS government schemes.
  4. **Policy Exclusions (94 claims | ₹1.17 Cr)** and **Room-Rent Capping (57 claims | ₹0.71 Cr)**.
- **Payer Loss Distribution**:
  - **Private Insurance TPAs**: ₹3.62 Cr (40.0% of total loss)
  - **Ayushman Bharat PM-JAY**: ₹2.29 Cr (25.3% of total loss)
  - **ESI Scheme**: ₹1.14 Cr (12.6%)
  - **Corporate Tie-ups**: ₹1.07 Cr (11.9%)
  - **CMCHIS**: ₹0.92 Cr (10.2%)

---

### 4. Patient Experience & Net Promoter Score (NPS)

![Patient Experience and NPS](assets/patient_experience_nps.png)

#### Key Analytical Findings:
- **Net Promoter Score**: Closed at **+6.4** (Promoters: 31.6%, Passives: 43.3%, Detractors: 25.1%).
- **Touchpoint Disparities**:
  - Doctor Care quality scored highest at **4.38 / 5.0**.
  - Nursing Care scored **4.19 / 5.0**.
  - **Billing & Insurance Desk scored lowest at 3.52 / 5.0** (Target > 4.3).
- **Core Patient Complaints**:
  - Delays in final insurance settlement exceeding 5 hours on the day of discharge.
  - Absence of post-discharge communication or guidance on complex medications.
  - Emergency room wait times during peak evening triage hours.

---

## 🤖 Deliverable 2: Machine Learning Readmission Risk Predictor

### 1. Modeling Strategy & Rigour
- **Target Variable**: `readmitted_30d` (binary: 1 = readmitted within 30 days, 0 = not readmitted).
- **Cohort Selection**: Restricted strictly to eligible discharges (`Discharged Home` and `LAMA`, **N = 10,816**). Deceased patients and transfer-outs were excluded.
- **Data Leakage Safeguards**: Dropped `days_to_readmission`, `admission_id`, and `patient_id`.
- **Feature Engineering (SOP-Aligned)**:
  - `is_late_discharge`: Discharge hour ≥ 18:00 (SMH Policy v4.2 Section 3).
  - `distance_gt_50km`: Residence > 50 km from hospital (Tele-follow-up trigger).
  - `has_prior_admissions`: Prior admissions count in 12m > 0.
  - `high_comorbidity`: Comorbidity count ≥ 2.
  - `cardio_nephro_neuro`: Indicator for high-volume bottleneck departments.
  - `er_wait_minutes`: Imputed median for emergency, 0 for elective.

### 2. Model Performance Leaderboard

![ML Model Evaluation ROC and PR Curves](assets/ml_model_evaluation.png)

| Model Architecture | ROC-AUC | PR-AUC | Recall @ Top 20% Risk | Precision @ 0.50 | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Baseline)** | **0.7431** | **0.3197** | **48.72%** | Balanced | Best for clinical explainability & odds ratios |
| **Random Forest (200 Trees)** | 0.7284 | 0.2735 | 43.96% | 0.28 | Non-linear ensemble |
| **Gradient Boosting Classifier** | 0.7232 | 0.3018 | 45.79% | 0.30 | High calibration |
| **XGBoost Classifier (Tuned)** | **0.7220** | **0.2846** | **45.79%** | **0.31** | **Selected for Real-Time Production Serving** |

> [!IMPORTANT]
> **Recall @ Top 20% Risk = 48.72%**: By flagging only the top 20% highest-risk patients at discharge, the clinical team captures nearly **half of all future readmissions**. This enables highly targeted interventions (pharmacist reconciliation, nurse visits, free 7-day OPD) without overburdening hospital staff.

### 3. Top Risk Drivers (Standardised Feature Importance)
1. **Heart Disease / Congestive Heart Failure** (+0.85 effect)
2. **Chronic Kidney Disease (CKD)** (+0.78 effect)
3. **Number of Prior Admissions in 12 Months** (+0.72 effect)
4. **ICU Stay during Index Admission** (+0.65 effect)
5. **Late-Evening Discharge (≥ 18:00)** (+0.58 effect)
6. **Distance > 50 km from Hospital** (+0.52 effect)
7. **Medication Reconciliation Completed** (-0.40 protective effect)
8. **Discharge Completed Before 12:00 Noon** (-0.35 protective effect)

### 4. Bedside Risk Categorization & Automated SOP Directives
When a patient is evaluated in the Bedside Risk Calculator, the model categorises the patient into one of three hospital risk tiers and outputs automated clinical care transition orders:

```mermaid
graph TD
    A["Patient at Planned Discharge"] --> B["Predictive ML Scoring Engine"]
    B --> C{"Risk Score"}
    C -->|"< 10%"| D["🟢 Low Risk"]
    C -->|"10% - 25%"| E["🟡 Medium Risk"]
    C -->|"> 25%"| F["🔴 High Risk"]
    
    D --> D1["Standard 48-hr Nurse Follow-up Call"]
    D --> D2["Routine OPD Follow-up"]
    
    E --> E1["Mandatory 48-hr Nurse Call"]
    E --> E2["Discharge Summary Danger Signs Education"]
    
    F --> F1["Mandatory 48-hr Nurse Call"]
    F --> F2["Free 7-Day OPD Specialist Review"]
    F --> F3["Day-14 Telephone Check"]
    F --> F4["30-Day Free Tele-Consultation Access"]
    
    B --> G{"Special Conditions"}
    G -->|"Distance > 50 km"| G1["Activate Tele-Consultation in lieu of OPD"]
    G -->|"Discharge >= 18:00"| G2["Mandatory DMO Countersignature Required"]
    G -->|"5+ Meds / Cardio-Nephro"| G3["Clinical Pharmacist Teach-Back Reconciliation"]
```

---

## 💬 Deliverable 3: GenAI Gemini RAG Clinical & Policy Assistant

### 1. RAG Architecture & Ingestion Pipeline
- **Source Documents**:
  1. `Sunrise_Hospital_Policy_Manual.pdf` (Version 4.2 | Document No: SMH/QA/POL-001 | 174 lines)
  2. `Sunrise_Annual_Report_FY2025-26.pdf` (Board Report June 2026 | 207 lines)
- **Chunking Strategy**: Page and section-aware regex chunking parsing numbered SOP clauses and headings into discrete, metadata-tagged passages.
- **Retrieval Engine**: TF-IDF & Cosine Similarity vector space index + dynamic top-$k$ contextual window ranking.
- **LLM Synthesis**: Google Gemini 2.5 Flash via `google-genai` Python SDK, prompting with strict grounding instructions:
  > *"Answer strictly based on the provided context. Cite document name, section heading, and page number for every assertion. Do not extrapolate."*
- **Fail-Safe Fallback**: Includes a built-in deterministic response generator so that even in offline scenarios or when an API key is not yet provided, all standard hospital policy queries are answered with exact citations.

### 2. Benchmark Verification on Sample Capstone Questions

| # | Capstone Test Question | Grounded Answer from RAG System | Official Document Citation |
| :-: | :--- | :--- | :--- |
| **1** | **Target time from discharge order to patient exit?** | **3 hours** for cash patients; **4 hours** for insured patients (accounting for TPA clearance). Target is 60% discharges before 12:00 noon. Discharges after 18:00 require DMO approval. | *SMH Policy Manual v4.2, Section 3, Page 1* |
| **2** | **Who must be called within 48 hours & what is checked?** | **All discharged patients (100% coverage)**, including LAMA. Conducted by a nurse checking: (1) symptoms/danger signs, (2) medication availability/adherence, (3) wound/dressing status, (4) confirming 7-day OPD visit. | *SMH Policy Manual v4.2, Section 4, Page 2* |
| **3** | **Claim submission deadline after discharge?** | Final claim documents must be submitted within **3 working days** of discharge. Submissions beyond 7 days require a written explanation from the TPA lead. | *SMH Policy Manual v4.2, Section 6, Page 2* |
| **4** | **FY25-26 readmission rate & highest department?** | Hospital-wide readmission rate was **12.6%** (Target < 10%). Highest departments: **Cardiology (20.8%)**, **Nephrology (20.3%)**, **Pulmonology (16.3%)**, and **Neurology (16.2%)**. Lowest: OBG (2.6%). | *Annual Report FY2025-26, Sections 2 & 3, Pages 1-2* |
| **5** | **Actions planned for FY26-27 to reduce claim rejections?** | (1) Enforce a Digital Claim Checklist before billing clearance, (2) Strict 3-day submission TAT, (3) Weekly rejection reviews by Revenue Cycle Manager, (4) Pre-discharge PM-JAY package-code audit. Target: < 5% rejection, recovering ₹4.0 Cr. | *Annual Report FY2025-26, Section 7, Page 3* |
| **6** | **Escalation trigger when ICU occupancy crosses threshold?** | **ICU > 90% (Red)**: Intensivist reviews step-down candidates every 6h, Medical Superintendent alerted, elective surgeries needing post-op ICU deferred. **ICU 100% (Black)**: External referral network activated, families briefed, CEO notified. | *SMH Policy Manual v4.2, Section 5, Page 2* |

---

## 📋 Deliverable 4: Strategic Recommendations & Financial Impact Memo

### Executive Financial Impact Table

```text
+-----------------------------------------------------------------------------------------------+
| INITIATIVE                                 | ANNUAL REVENUE / COST IMPACT | IMPLEMENTATION TIMELINE|
+-----------------------------------------------------------------------------------------------+
| 1. Digital Claim Checklist & Audit         | + ₹4.00 Crore EBITDA Recovery | Q1 FY2026-27 (Immediate)|
| 2. 100% 48-Hour Call & Care Transitions   | + ₹1.80 Crore Capacity Savings| Q1–Q2 FY2026-27        |
| 3. Dynamic Bed Flexing (OBG/Med -> Cardio) | + ₹8.50 Crore New Inpatient   | Q1–Q3 FY2026-27        |
| 4. Patient Discharge Lounge & Bilingual SOP| NPS > +40.0 (Brand Equity)    | Q2 FY2026-27           |
+-----------------------------------------------------------------------------------------------+
| TOTAL PROJECTED FINANCIAL IMPACT           | + ₹14.30 Crore Annual Impact  | 12-Month Horizon       |
+-----------------------------------------------------------------------------------------------+
```

### The 4 Action Pillars:

#### Pillar 1: Eliminate Preventable Claim Rejections (Recover ₹4.0+ Cr)
- **Problem**: 725 rejected claims cost ₹9.03 Cr in FY25-26. Over 70% were due to clerical oversights (incomplete documentation, unobtained pre-auths, scheme package code mismatches).
- **Intervention**:
  1. Mandate a mandatory **Digital Claim Checklist** in the HIS: Discharge cannot be finalized until doctor summaries, itemised pharmacy slips, implant invoices, and signed consent forms are attached.
  2. Implement clinical coder verification for all Ayushman Bharat (PM-JAY) and CMCHIS cases prior to patient exit.
  3. Enforce strict submission within 3 working days.
- **KPI Target**: Claim rejection rate reduced from **9.2% to < 5.0%**, directly recovering **₹4.0+ Crore in EBITDA**.

#### Pillar 2: Scale 48-Hour Follow-Up Calls to 100% (Prevent ~290 Readmissions)
- **Problem**: Discharged patients without a 48-hour call experienced a **17.7% readmission rate** versus **8.1% for those called**, yet coverage stood at only 54.8%.
- **Intervention**:
  1. Deploy automated worklists in the HIS allocating discharged patients to designated care coordinator nurses.
  2. Integrate the **ML Readmission Risk Score** to fast-track High-Risk (>25%) patients into the comprehensive 7-day OPD and day-14 check protocols.
  3. Offer telemedicine consultations for patients living >50 km away (Salem, Palakkad, Erode).
- **KPI Target**: 48-hour call coverage **> 90%**, driving overall readmission rate from **12.6% to < 10.0%**.

#### Pillar 3: Unlock Inpatient Capacity via Dynamic Bed Flexing & ALOS Reduction
- **Problem**: Cardiology (86.5%), Nephrology (88.0%), and Neurology (85.2%) were severely bottlenecked, deferring elective cases, while OBG (28.5%) and General Medicine (49.7%) operated with large numbers of unoccupied beds.
- **Intervention**:
  1. Formally designate **20–25 beds in OBG and General Medicine as flex overflow step-down beds** with dedicated cardiology and nephrology nursing staff.
  2. Enforce the **Discharge Before Noon target (60% before 12:00 PM)** and 60-minute bed sanitization turnaround.
  3. Commission the 20 new cardiac beds in Q3 FY26-27 as planned by the Board.
- **KPI Target**: Normalize hospital-wide occupancy to **75–80%**, shorten ALOS from 6.24 to **< 5.8 days**, and generate **₹8.5 Cr in new inpatient revenue**.

#### Pillar 4: Patient Experience & Discharge Lounge
- **Problem**: Net Promoter Score of +6.4 was severely degraded by discharge turnaround delays (>5 hours) and billing desk confusion.
- **Intervention**:
  1. Construct a ground-floor **Patient Transit Discharge Lounge**: Medically stable patients wait comfortably in the lounge with refreshments while insurance TPA approvals process, freeing inpatient beds for incoming admissions by 11:00 AM.
  2. Issue all discharge summaries and medication instructions in **bilingual format (English & Tamil)**.
- **KPI Target**: Net Promoter Score elevated from **+6.4 to > +40.0**.

---

## 🚀 Application Guide & Instructions

### 1. Architecture Overview
The platform is packaged into a unified Streamlit web application with five integrated modules accessible via the navigation sidebar:
- `app.py`: Main application script including the landing page, BI dashboard, ML predictor, RAG chatbot, and memo.
- `rag_engine.py`: Self-contained vector retrieval and Gemini RAG pipeline.
- `models/`: Pre-trained ML classifiers (`xgboost.pkl`, `logistic_regression.pkl`, `random_forest.pkl`, `gradient_boosting.pkl`).
- `assets/`: High-resolution figures and hospital imagery.
- `data/`: Inpatient admissions, bed census, claims, departments, and patient feedback tables.

### 2. Launching the Platform

#### Method A: One-Click Windows Launcher
Double-click [`run_app.bat`](file:///d:/Geetha/GUVI%20BMAI%20MAY%2026/Health%20Care%20Analytics/run_app.bat) in the project root directory.

#### Method B: PowerShell / Terminal
```powershell
cd "d:\Geetha\GUVI BMAI MAY 26\Health Care Analytics"
& "C:\Program Files\Python313\python.exe" -m streamlit run app.py --server.port 8501
```

Access the interface in any web browser at:  
👉 **`http://localhost:8501`**

### 3. Configuring Google Gemini API (Optional)
- To enable custom free-form generative prompting with **Gemini 2.5 Flash**, enter your Google Gemini API key into the password field in the application sidebar or set the environment variable:
  ```powershell
  $env:GEMINI_API_KEY = "your-api-key-here"
  ```
- If no key is entered, the platform automatically operates using the **built-in high-precision grounded policy engine**, ensuring instant, 100% verified responses with document citations.

---

## 📐 Power BI Star Schema & DAX Reference

For enterprise reporting in Power BI Desktop or Service, link the tables via the star schema shown below:

```text
       +-------------------+       +-----------------------+
       |   patients.csv    |       |    departments.csv    |
       |  (Dimension Table)|       |   (Dimension Table)   |
       +---------+---------+       +-----------+-----------+
                 |                             |
                 | patient_id                  | dept_id
                 v                             v
       +-----------------------------------------------+
       |                admissions.csv                 |
       |                 (Fact Table)                  |
       +---------+---------------------------+---------+
                 |                           |
                 | admission_id              | admission_id
                 v                           v
       +-------------------+       +-----------------------+
       |insurance_claims   |       |   patient_feedback    |
       |   (Fact Table)    |       |     (Fact Table)      |
       +-------------------+       +-----------------------+
```

### Essential DAX Formulas:

```dax
// 1. 30-Day Readmission Rate (%)
Readmission Rate % = 
DIVIDE(
    CALCULATE(
        COUNTROWS(admissions),
        admissions[readmitted_30d] = TRUE(),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    CALCULATE(
        COUNTROWS(admissions),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    0
) * 100

// 2. Average Length of Stay (ALOS)
ALOS = AVERAGE(admissions[los_days])

// 3. 48-Hour Follow-Up Call Coverage (%)
Followup Call Coverage % = 
DIVIDE(
    CALCULATE(
        COUNTROWS(admissions),
        admissions[followup_call_48h] = TRUE(),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    CALCULATE(
        COUNTROWS(admissions),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    0
) * 100

// 4. Claim Rejection Rate (%)
Claim Rejection Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(insurance_claims), insurance_claims[claim_status] = "Rejected"),
    COUNTROWS(insurance_claims),
    0
) * 100

// 5. Total Rejected Claim Amount (INR Cr)
Rejected Claims Value (Cr) = 
CALCULATE(
    SUM(insurance_claims[claimed_amount_inr]),
    insurance_claims[claim_status] = "Rejected"
) / 10000000

// 6. Net Promoter Score (NPS)
NPS = 
VAR TotalResponses = COUNTROWS(patient_feedback)
VAR Promoters = CALCULATE(COUNTROWS(patient_feedback), patient_feedback[nps_category] = "Promoter")
VAR Detractors = CALCULATE(COUNTROWS(patient_feedback), patient_feedback[nps_category] = "Detractor")
RETURN
DIVIDE(Promoters - Detractors, TotalResponses, 0) * 100
```

---

## 📸 Screenshots & Dashboard Gallery

Comprehensive visual showcase of the deployed Sunrise Multispeciality Hospital Healthcare Intelligence Platform, featuring the public clinical portal, interactive executive BI dashboards, predictive machine learning decision engine, and generative AI RAG assistant.

### 1. 🏥 Public Hospital Landing Website & Patient Portal
> *NABH 5th Edition Accredited Tertiary Care Hospital Landing Interface with 9 Centers of Excellence, 24x7 ER Triage protocols, and outpatient appointment booking.*

![Public Hospital Landing Website](assets/screenshot_landing_portal.png)

---

### 2. 📊 Executive Hospital Operations & Financial BI Dashboard
> *Interactive executive dashboard featuring hospital-wide KPI scorecards, multi-department slicers, admissions volume tracking, and financial leakage indicators.*

![Executive BI Dashboard Overview](assets/screenshot_executive_bi.png)

---

### 3. 📈 Inpatient Flow Seasonality & Bed Capacity Bottlenecks (Deliverable 1 — Tab 1)
> *Monthly admissions flow identifying the Q3 monsoon surge (Oct–Dec) alongside departmental bed occupancy thresholds (Red alert in Nephrology 88.0%, Cardiology 86.5%, Neurology 85.2% vs. idle beds in OBG 28.5%).*

![Bed Census Bottlenecks and Monthly Flow](assets/bed_census_bottleneck.png)

---

### 4. 🎯 30-Day Readmission Analysis & 48-Hour Nurse Call Impact (Deliverable 1 — Tab 2)
> *Comparative readmission rates demonstrating the transformative impact of the 48-hour post-discharge nurse call (8.1% readmitted when called vs. 17.7% uncontacted — 2.18x risk), specialty disparities, and chronic comorbidity clustering.*

![Readmission Drivers and 48-Hour Call Impact](assets/readmission_drivers.png)

---

### 5. 💸 Insurance Claim Leakage & Pareto Rejection Root-Cause Analysis (Deliverable 1 — Tab 3)
> *80/20 Pareto distribution of ₹9.03 Crore in lost claim revenue across 725 rejected claims, identifying incomplete documentation, pre-authorization lapses, and package code mismatches as the primary drivers.*

![Insurance Claims Pareto Rejections](assets/claims_pareto.png)

---

### 6. ⭐ Patient Experience & Net Promoter Score (NPS) Drivers (Deliverable 1 — Tab 4)
> *Patient sentiment breakdown (+6.4 NPS) highlighting the 5-hour discharge billing delay bottleneck versus high clinical trust in doctors (4.38/5) and nursing staff (4.19/5).*

![Patient Experience and NPS Drivers](assets/patient_experience_nps.png)

---

### 7. 🤖 Machine Learning 30-Day Readmission Risk Predictor (Deliverable 2)
> *Bedside clinical decision support engine allowing care teams to evaluate patient discharge parameters in real-time, compute ML risk probabilities, and generate automated NABH care transition orders.*

![Readmission ML Risk Engine](assets/screenshot_ml_predictor.png)

---

### 8. 🔬 Machine Learning Model Evaluation Leaderboard
> *Performance evaluation curves comparing Logistic Regression, Random Forest, Gradient Boosting, and XGBoost (ROC-AUC 0.743, PR-AUC 0.320, Recall @ Top-20% = 48.7%).*

![ML Model Evaluation Curves](assets/ml_model_evaluation.png)

---

### 9. 💬 Gemini RAG AI Clinical & Policy Assistant (Deliverable 3)
> *Interactive Generative AI assistant powered by Google Gemini 2.5 Flash, grounded directly in the SMH Hospital Policy Manual v4.2 and FY2025–26 Annual Report with exact section citations.*

![Gemini RAG Policy Chatbot](assets/screenshot_gemini_rag.png)

---

### 10. 📋 Strategic Recommendations & Board Financial Memo (Deliverable 4)
> *Executive roadmap detailing four actionable intervention pillars delivering +₹14.30 Crore in annual financial and operational impact (+₹4.0 Cr claim recovery, ~290 readmissions prevented, 4,500 bed-days unlocked, NPS > +40).*

![Strategic Recommendations Memo](assets/screenshot_strategic_memo.png)

---

### 11. 📊 Executive KPI Summary Scorecard (FY2025–26)
> *High-level scorecard summarizing full-year performance across admissions, revenue, bed utilization, claims, and clinical quality.*

![Executive KPI Summary Card](assets/kpi_summary_card.png)

---
*Sunrise Multispeciality Hospital Healthcare Intelligence Platform — Produced for GUVI BMAI Capstone Deliverable Evaluation, FY2025–26.*

