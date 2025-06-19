#These are just mappings

#Divides ACI classes into attack categories
ACI_CATEGORY_MAPPING = {
    0: "Benign",
    1: "Reckoning",
    2: "Reckoning",
    3: "Reckoning",
    4: "DOS",
    5: "DOS",
    6: "DOS",
    7: "DOS",
    8: "DOS",
    9: "Brute Force",
}

#Provides mapping for 
ACI_PROPORTION_MAPPING = {
    "Benign": {"Benign": 0.3, "Reckoning": 0.3, "DOS": 0.3, "Brute Force": 0.1},
    "Reckoning": {"Reckoning": 0.3, "DOS": 0.4, "Brute Force": 0.2, "Benign": 0.2},
    "DOS": {"DOS": 0.4, "Brute Force": 0.3, "Reckoning": 0.2, "Benign": 0.2},
    "Brute Force": {"Brute Force": 0.25, "DOS": 0.25, "Reckoning": 0.25, "Benign": 0.25},
}


#Provides mapping for when the labels are numerical - main code uses label encoder usually.
ACI_CLASS_MAP = {'Benign': 0,
 'OS Scan': 1,
 'Vulnerability Scan': 2,
 'Port Scan': 3,
 'ICMP Flood': 4,
 'Slowloris': 5,
 'SYN Flood': 6,
 'UDP Flood': 7,
 'DNS Flood': 8,
 'Dictionary Attack': 9}