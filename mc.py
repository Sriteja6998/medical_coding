
import os
import json
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

bedrock= ""
inference_profile_arn=""

def call_claude(input: str) -> str:
    body = json.dumps({
        "max_tokens": 40000,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "user", "content": input}
        ],
        "anthropic_version": "bedrock-2023-05-31"
    })
    response = bedrock.invoke_model(body=body, modelId=inference_profile_arn)
    response_body = json.loads(response['body'].read())
    print(f"Input tokens: {response_body['usage']['input_tokens']}")
    print(f"output tokens: {response_body['usage']['output_tokens']}")
    return response_body.get("content")[0].get("text")


# text = """
# Chief Complaint
# “My stomach hurts and I feel full of gas.”
# History
# 47 year old male with mid-abdominal epigastric pain1, associated with severe nausea & vomiting; unable to keep down any food or liquid. Pain has become “severe” and constant.
# Has had an estimated 13 pound weight loss over the past month.
# Patient reports eating 12 sausages at the Sunday church breakfast five days ago which he believes initiated his symptoms.
# Patient admits to a history of alcohol dependence2. Consuming 5 – 6 beers per day now, down from 10 – 12 per day 6 months ago. States that he has nausea and sweating with “the shakes” when he does not drink.
# Exam
# VS: T 99.8°F, otherwise normal.
# Mild jaundice noted.
# Abdomen distended and tender across upper abdomen3. Guarding is present. Bowel sounds diminished in all four quadrants.
# Oral mucosa dry, chapped lips, decreased skin turgor
# Assessment and Plan
# Dehydration and suspected acute pancreatitis.
# Admit to the hospital. Orders written and sent to on-call hospitalist.
# 1L IV NS started in office. Blood drawn for labs.
# Recommend behavioral health counseling for substance abuse assessment and possible treatment.
# Patient’s wife notified of plan; she will transport to hospital by private vehicle.
# Summary of ICD-10-CM Impacts
# Clinical Documentation
# Describe the pain as specifically as possible based on location.
# When addressing alcohol related disorders you should distinguish alcohol use, alcohol abuse, and alcohol dependence. ICD-10-CM has changed the terminology and the parameters for coding substance abuse disorders. In this encounter note, as the acute pancreatitis is suspected, and the patient’s alcohol intake status is stated, the associated alcoholism code is listed.
# Abdominal tenderness may be coded. Ideally the documentation should include right or left upper quadrant and indicate if there is rebound in order to identify a more specific code. Currently the ICD-10 code would be R10.819, Abdominal tenderness, unspecified site as the documentation is insufficient in laterality and specificity.
# """

# text = """
# Chief Complaint
# “I’m here for my annual check-up.1”
# History
# 73 year old male with history of coronary artery disease, stent placement, hyperlipidemia, HTN and GERD.
# Recent admission to hospital following a hypertensive crisis. Discharged home on olmesartan medoxomil 20 mg daily.
# Patient stopped taking olmesartan medoxomil due to side effects2, including a headache that began after starting the medication and still exists, and tiredness.
# Regular activity includes walking, golfing. Active social life. No complaints of chest pain, or dyspnea on exertion.
# Last colonoscopy was 9 months ago. No significant pathology found; some diverticular disease.
# Medications were reviewed.
# Exam
# Chest clear. Heart sounds normal. Mental status exam intact.
# EKG shows no changes from prior EKG.
# Vitals: BP is 159/95, otherwise normal. Per patient, he had good control of BP on meds, but it has risen without medication.
# BUN/creatinine normal limits.
# Assessment and Plan
# HTN noted on exam today. Change from olmesartan medoxomil to metoprolol tartrate 50 mg once daily, will titrate dosage every two weeks until BP normalizes.
# Discussed the importance of daily home BP monitoring, low sodium diet, and taking BP medication as prescribed; he verbalizes understanding.
# Schedule follow-up visit in two weeks to evaluate effectiveness of new BP medication therapy, and repeat BUN/creatinine.
# Summary of ICD-10-CM Impacts
# Clinical Documentation
# Documenting why the encounter is taking place is important, as the coder may assign a different code based on the type of visit (e.g., screening, with no complaint or suspected diagnosis, for administrative purposes). In this situation, the patient is requesting an encounter without a complaint, suspected or reported diagnosis.
# Document that the patient is noncompliant with his medication. This “underdosing” concept can often be coded, along with the patient’s reason for not taking the prescribed medications. Document if there is a medical condition linked to the underdosing that is relevant to the encounter, and ensure the connection is clearly made. The ICD-10-CM terms provide new detail as compared to the ICD-9-CM code V15.81, history of past noncompliance. In this case there was no noted history of noncompliance. In this note the side effects of stopping the medication include headache, which remains as a patient complaint for this encounter. When documenting headache do differentiate if intractable versus non-intractable.
# """

text = """
S:Mrs. Finley presents today after having a new cabinet fall on her last week, suffering a concussion, as well as some cervicalgia. She was cooking dinner at the home she shares with her husband. She did not seek treatment at that time. She states that the people that put in the cabinet in her kitchen missed the stud by about two inches. Her husband, who was home with her at the time told her she was "out cold" for about two minutes. The patient continues to have cephalgias since it happened, primarily occipital, extending up into the bilateral occipital and parietal regions. The headaches come on suddenly, last for long periods of time, and occur every day. They are not relieved by Advil. She denies any vision changes, any taste changes, any smell changes. The patient has a marked amount of tenderness across the superior trapezius.

O:Her weight is 188 which is up 5 pounds from last time, blood pressure 144/82, pulse rate 70, respirations are 18. She has full strength in her upper extremities. DTRs in the biceps and triceps are adequate. Grip strength is adequate. Heart rate is regular and lungs are clear.

A:1. Status post concussion with acute persistent headaches
2. Cervicalgia
3. Cervical somatic dysfunction
P:The plan at this time is to send her for physical therapy, three times a week for four weeks for cervical soft tissue muscle massage, as well as upper dorsal. We’ll recheck her in one month, sooner if needed.
"""

instruction ="""
You are a certified medical coding assistant. Your task is to read unstructured clinical documentation and extract standardized billing codes. 

Your output must include:
1. **ICD-10** codes for all diagnoses mentioned (including primary and secondary diagnoses)
2. **CPT** codes for all documented procedures, tests, and consultations
3. **HCPCS** codes for any medications, equipment, or services not covered by CPT
4. **Modifiers** if applicable (e.g., bilateral, repeated procedure)
5. A **confidence score** (1-5) for each extracted code
6. A **brief justification** for why each code was selected

---

### Guidelines:

- Use only official **ICD-10**, **CPT**, and **HCPCS Level II** codes as per AMA and CMS guidelines.
- Include **only codes that are explicitly supported by clinical evidence in the note**.
- If the documentation is insufficient to assign a code with confidence, mention `"confidence": 1` and add `"description": "Not enough clinical detail"` as justification.
- Do not hallucinate or assume any procedures or diagnoses not mentioned.
- Codes must be **justifiable in a real medical audit**.

---

Ensure the output is valid HTML and includes **all required fields** for downstream billing, analytics, and audit logging.

### Clinical Context (Input):


"""

# instruction ="""
# Given the following medical note, extract:
# 1. ICD-10 diagnosis codes
# 2. CPT procedure codes
# 3. HCPCS codes if any

# Generate a concise summary of the medical note, focusing on the key clinical details and coding implications.
# Return output as HTML.

# Medical Note:

# """
response = call_claude(text)

print(response)
