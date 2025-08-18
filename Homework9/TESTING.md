# Manual Verification & Testing

This document outlines the steps to manually verify the functionality of the end-to-end pipeline. The tests check both the machine learning model's predictive capability and the generative AI's prescriptive output.

### Prerequisites
Before running these tests, ensure you have completed all setup steps from the main `README.md`, including:
1. A working Python environment with all dependencies installed.
2. The `GOOGLE_API_KEY` environment variable set.
3. The DGA detection model (`DGA_Leader.zip`) trained and exported by running `python 1_train_and_export.py`.

---

### Test 1: Legitimate Domain
This test verifies that the model correctly identifies a benign, legitimate domain.

**Command:**
```bash
python 2_analyze_domain.py --domain google.com
```

**Expected Output:**
The script should classify the domain as **legit**. The final output should be a simple message indicating the domain is not a DGA, and the prescriptive playbook generation will be skipped.

```
Analyzing domain: google.com...
Domain is legit.
No DGA detected. Analysis complete.
```

---

### Test 2: DGA Domain
This test verifies that the model correctly identifies a DGA-like domain and that the generative AI component successfully creates a prescriptive playbook.

**Command:**
```bash
python 2_analyze_domain.py --domain d6w8x4z1s9q.net
```

**Expected Output:**
The script should classify the domain as **dga**. It will then generate a prescriptive playbook. The final output will be a JSON object containing a `summary`, `risk_level`, `recommended_actions`, and `communication_draft`.

```
Analyzing domain: d6w8x4z1s9q.net...
Domain is dga.
Generating prescriptive playbook...
{
  "summary": "Suspicious domain 'd6w8x4z1s9q.net' with high entropy detected, likely associated with a DGA. Immediate action is required to prevent a potential malware infection or data exfiltration.",
  "risk_level": "Critical",
  "recommended_actions": [
    "Block 'd6w8x4z1s9q.net' at the firewall and DNS level immediately.",
    "Search security logs for any internal systems that have communicated with this domain.",
    "Isolate any host that has accessed the domain for further forensic analysis.",
    "Update threat intelligence feeds and security appliances with the new indicator of compromise."
  ],
  "communication_draft": "We have identified a potential security threat associated with a malicious domain. Please refrain from accessing or clicking any links related to 'd6w8x4z1s9q.net'."
}
