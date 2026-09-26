# Error Analysis — Amazon ML Challenge 2026

**Model**: LogisticRegression | **Threshold**: 0.60 | **Macro F0.5**: 0.8911

## 1. False Positive Categories

| Category | Count | Percentage |
|---|---:|---:|
| generic_name_collision | 103 | 48.1% |
| same_name_diff_address | 59 | 27.6% |
| address_number_conflict | 34 | 15.9% |
| same_address_diff_business | 16 | 7.5% |
| abbreviation_collision | 2 | 0.9% |

### Sample False Positives:

- **Category**: generic_name_collision (Prob: 0.737)
  - S1: `Bangalore South Electronics Private Limited` | Addr: `No. 79/1, Aishwarya Sampurna Building, 4Th Floor, Vani Vilas Road, Bangalore South, Bangalore, Karnataka`
  - Target: `Bangalore South Effects  Private Limited` | Addr: `3788, Bangalore South, Bangalore, KA`

- **Category**: same_name_diff_address (Prob: 0.945)
  - S1: `Jai Industries Private Limited` | Addr: `C/O. Hem Chand Dhariwal, Shubh, Gokhley Near Bank Colony, Policeline 1Stcrossing, Ajmer, Rajasthan`
  - Target: `Jai Interior-Private Limited` | Addr: `None`

- **Category**: same_name_diff_address (Prob: 0.851)
  - S1: `Hari Business Private Limited` | Addr: `Tower3-304, Manglamaananda, Dada Gurudev Nagar, Sanganer, Jaipur, Rajasthan`
  - Target: `Hari Business Pcae Limited` | Addr: `None`

- **Category**: address_number_conflict (Prob: 0.944)
  - S1: `Toney Athena PLLC` | Addr: `9249 Crossbill Drive, Leland, NC`
  - Target: `Toney Athena Southside PLLC` | Addr: `9254 Crossbill Drive, Leland, North Carolina`

- **Category**: same_name_diff_address (Prob: 0.877)
  - S1: `Chiropractic Care Associates` | Addr: `8401 14th Avenue, Bloomington, MN`
  - Target: `Chiropractic Care Associates of` | Addr: `None`

## 2. False Negative Categories

| Category | Count | Percentage |
|---|---:|---:|
| missing_address | 92 | 34.7% |
| address_variation | 91 | 34.3% |
| trade_name_or_transliteration | 40 | 15.1% |
| severe_typo_or_abbrev | 38 | 14.3% |
| missing_name | 4 | 1.5% |

### Sample False Negatives:

- **Category**: address_variation (Prob: 0.317)
  - S1: `Green Healthcare Private Limited` | Addr: `Plot No B352, Second Floor Front Green Field Colony, Faridabad, Haryana`
  - Target: `Green Healthcare Private  Limited` | Addr: `HR, Faridabad, Plot No Bg-352, Divreportingcircle`

- **Category**: missing_address (Prob: 0.151)
  - S1: `Bernetta Sanchez Mountain Restaurant` | Addr: `115 Hathaway Heights Road, Anniston, AL`
  - Target: `Bernetta Sanchez Móuntain Center` | Addr: `None`

- **Category**: missing_address (Prob: 0.573)
  - S1: `Pediatric Keystone Health Inc.` | Addr: `Harlingen, 814 Sul Ross Avenue, TX`
  - Target: `Pediatric Kyetoe Health Inc.` | Addr: `None`

- **Category**: missing_address (Prob: 0.107)
  - S1: `Zaify Park Group` | Addr: `551 Goose Crossing, Farmington, AR`
  - Target: `ZAIFY GROUP CENTER` | Addr: `None`

- **Category**: missing_address (Prob: 0.586)
  - S1: `New Ricco Industrial Area Riders Private Limited` | Addr: `Chittorgarh, F-116, Rajasthan, New Ricco Industrial Area`
  - Target: `New Ricco Industrial Riders Private Limited Services` | Addr: `None`

