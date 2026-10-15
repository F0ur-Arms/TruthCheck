"""
Part 3: Claims 55 to 82
Covers:
- Organ health & Emergency first aid myths (55-57)
- South Asian Diet & Food Myths (58-71)
- Food Combinations & Digestion Myths (72-80)
- Cancer & Radiation food myths (81-82)
"""

PART3_VARIATIONS = {
    # 55. liver detox supplements
    "liver detox supplements": [
        {
            "claim_subject": "endogenous hepatic detoxification mechanisms",
            "claim_text": "The human liver naturally and continuously neutralizes and eliminates toxins through Phase I (cytochrome P450) and Phase II conjugation pathways without requiring commercial detox supplements.",
            "verdict": "TRUE",
            "scientific_truth": "The liver, along with the kidneys, lungs, and gastrointestinal tract, possesses complex enzymatic systems that convert lipophilic xenobiotics into water-soluble metabolites for biliary and urinary excretion; commercial 'detox' teas provide no clinical enhancement of this physiology.",
            "risk_level": "Low",
            "medical_domain": "hepatology",
            "population": "general population",
            "source_title": "Hepatic Drug Metabolism and Detoxification: Goodman & Gilman",
            "evidence_type": "educational_content",
            "evidence_summary": "Medical pharmacology establishes that hepatic enzymatic biotransformation is continuous and unassisted by over-the-counter herbal cleanses.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "detox supplements flush 10 pounds of liver sludge in 24 hours",
            "claim_text": "Drinking an olive oil and lemon juice liver cleanse expels dozens of toxic green liver stones and completely cures fatty liver disease in 24 hours.",
            "verdict": "FALSE",
            "scientific_truth": "The 'stones' passed in stool after olive oil flushes are merely soap complexes (saponified oil globules formed by gastric acid and potassium), not true gallstones or hepatic calculi; high oil ingestion can precipitate acute biliary pancreatitis.",
            "risk_level": "High",
            "medical_domain": "hepatology",
            "population": "general population",
            "source_title": "Could Gallbladder Flushes Really Expel Gallstones? Lancet Case Report",
            "evidence_type": "expert_consensus",
            "evidence_summary": "Lancet chemical analyses demonstrate that expelled 'gallstones' from flushes are chemically identical to olive oil saponification products formed in the gut.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "silymarin modest antioxidant properties in chronic liver disease",
            "claim_text": "Milk thistle extract (silymarin) shows modest antioxidant and anti-fibrotic activity in laboratory settings, but clinical trials show inconsistent benefits for reversing cirrhosis.",
            "verdict": "PARTLY_TRUE",
            "scientific_truth": "Silymarin exhibits hepatoprotective membrane-stabilizing effects in preclinical models, but large randomized multicenter trials have not proven significant histological reversal of advanced hepatic fibrosis in humans.",
            "risk_level": "Low",
            "medical_domain": "hepatology",
            "population": "liver disease patients",
            "source_title": "Milk Thistle for Alcoholic and/or Hepatitis B or C Liver Diseases: Cochrane Review",
            "evidence_type": "systematic_review",
            "evidence_summary": "Cochrane reviews find no convincing evidence that milk thistle reduces mortality or improves liver histology in viral or alcoholic liver disease.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 56. snake bite suction
    "snake bite suction": [
        {
            "claim_subject": "invalidation of suction and tourniquets in snake envenomation",
            "claim_text": "Cutting the wound or using mouth suction on a venomous snakebite is medically contraindicated and increases tissue necrosis, infection, and severe bleeding.",
            "verdict": "TRUE",
            "scientific_truth": "Venom spreads rapidly through lymphatic channels within seconds. Incising the bite site damages tendons and nerves, introduces oral bacteria, and extracts less than 1% of venom, while tourniquets cause limb-threatening ischemia.",
            "risk_level": "Low",
            "medical_domain": "emergency_medicine",
            "population": "general population",
            "source_title": "WHO Guidelines for the Management of Snakebites",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "WHO and Wilderness Medical Society guidelines condemn incising, sucking, or applying tight arterial tourniquets for snakebites.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "sucking snake venom out of wound cures venomous bite completely",
            "claim_text": "If bitten by a venomous cobra, slicing an X over the fang marks and sucking the venom with your mouth saves your life and avoids needing antivenom.",
            "verdict": "FALSE",
            "scientific_truth": "This movie myth is lethal. Venom cannot be extracted by oral suction, and the rescuer risks poisoning their own oral mucosa while the victim bleeds and delays definitive antivenom therapy in a hospital.",
            "risk_level": "High",
            "medical_domain": "emergency_medicine",
            "population": "general population",
            "source_title": "Suction Devices and Cutting for Snakebite Envenomation: Annals of Emergency Medicine",
            "evidence_type": "systematic_review",
            "evidence_summary": "Emergency medicine trials prove commercial suction extractors and mouth suction remove clinically negligible venom (<0.04%) and cause localized mechanical trauma.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "pressure immobilization technique for neurotoxic elapid bites",
            "claim_text": "Applying a broad, firm elastic bandage over the entire bitten limb and immobilizing it with a splint retards lymphatic flow and delays systemic neurotoxin absorption.",
            "verdict": "TRUE",
            "scientific_truth": "Pressure-immobilization bandage (50-70 mmHg, similar to a sprained ankle wrap) significantly slows lymphatic transit of high-molecular-weight elapid neurotoxins while awaiting hospital antivenom delivery.",
            "risk_level": "Low",
            "medical_domain": "emergency_medicine",
            "population": "snakebite victims",
            "source_title": "Pressure Immobilization for Snakebite: Evidence-Based Emergency Care",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Australian and international guidelines recommend pressure immobilization for neurotoxic elapid envenomation to slow systemic venom dissemination.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 57. heart attack cough CPR
    "heart attack cough CPR": [
        {
            "claim_subject": "cough cpr inefficacy and danger during acute myocardial infarction",
            "claim_text": "Coughing repeatedly does not treat an acute myocardial infarction and should never replace calling emergency medical services and chewing aspirin.",
            "verdict": "TRUE",
            "scientific_truth": "A heart attack is caused by coronary artery thrombosis obstructing myocardial blood flow. Coughing does not recanalize the occluded coronary vessel and causes physical exhaustion, tachycardia, and increased myocardial oxygen demand.",
            "risk_level": "Low",
            "medical_domain": "emergency_medicine",
            "population": "general population",
            "source_title": "AHA Statement: Cough CPR and Sudden Cardiac Arrest",
            "evidence_type": "expert_consensus",
            "evidence_summary": "American Heart Association strongly advises against 'cough CPR' for the general public, confirming it has no role in managing heart attacks.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "vigorous coughing cures acute heart attack instantly without hospital",
            "claim_text": "If you feel a heart attack coming on, coughing vigorously every two seconds dissolves the heart clot and cures you without calling an ambulance.",
            "verdict": "FALSE",
            "scientific_truth": "Following this viral social media hoax delays emergency reperfusion (PCI or thrombolysis), resulting in extensive transmural myocardial necrosis or fatal out-of-hospital cardiac arrest.",
            "risk_level": "High",
            "medical_domain": "emergency_medicine",
            "population": "cardiovascular patients",
            "source_title": "Management of Acute Coronary Syndromes: ESC Guidelines",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "European Society of Cardiology highlights rapid EMS activation as the single most critical determinant of survival in acute coronary syndromes.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "monitored rhythmic coughing in hospital cardiac catheterization labs",
            "claim_text": "In a hospital catheterization lab under continuous ECG monitoring, a conscious patient experiencing sudden documented ventricular arrhythmia may be instructed to cough to maintain brief cerebral perfusion before defibrillation.",
            "verdict": "PARTLY_TRUE",
            "scientific_truth": "Intense rhythmic coughing generates abrupt intrathoracic pressure spikes that temporarily maintain cerebral blood flow for a few seconds during witnessed, monitored ventricular asystole or tachycardia while the defibrillator is charged.",
            "risk_level": "Medium",
            "medical_domain": "emergency_medicine",
            "population": "hospitalized monitored patients",
            "source_title": "Cough-Induced Cardiac Compression: Experimental Observations in Catheterization Laboratories",
            "evidence_type": "systematic_review",
            "evidence_summary": "Cardiology literature acknowledges specialized utility strictly for monitored, conscious patients in procedural suites, not for laypeople experiencing chest pain at home.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 58. burnt roti cancer
    "burnt roti cancer": [
        {
            "claim_subject": "acrylamide and polycyclic aromatic hydrocarbons in burnt food",
            "claim_text": "Charred flatbreads contain traces of acrylamide and polycyclic aromatic hydrocarbons, but eating an occasional burnt spot on a roti does not significantly increase human cancer risk.",
            "verdict": "TRUE",
            "scientific_truth": "While high-temperature Maillard reactions produce acrylamide in starches, the concentrations in occasionally charred flatbreads are far below toxicological thresholds established in rodent bioassays, and human epidemiological studies show no conclusive cancer link.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Dietary Acrylamide and Cancer Risk: An Updated Meta-Analysis",
            "evidence_type": "systematic_review",
            "evidence_summary": "Systematic epidemiological meta-analyses find no statistically significant association between typical dietary acrylamide consumption and human cancer incidence.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "one burnt roti causes instant fatal colon cancer",
            "claim_text": "Eating a single roti with a black charred edge injects lethal carcinogens that trigger immediate malignant tumors throughout your digestive tract.",
            "verdict": "FALSE",
            "scientific_truth": "This is an extreme exaggeration. The body has enzymatic glutathione and aldehyde dehydrogenase defenses that neutralize low-level dietary Maillard reaction byproducts without generating spontaneous malignancies.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Acrylamide and Cancer Risk: American Cancer Society",
            "evidence_type": "expert_consensus",
            "evidence_summary": "ACS clarifies that while avoiding heavily burnt foods is prudent culinary practice, occasional consumption of charred spots poses negligible clinical risk.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "direct flame cooking versus heat exposure minimization",
            "claim_text": "Cooking flatbreads on a tawa (griddle) rather than directly over open gas burner flames reduces the deposition of unburnt hydrocarbons and heterocyclic compounds.",
            "verdict": "TRUE",
            "scientific_truth": "Cooking on an indirect griddle reduces soot and combustion byproduct deposition compared to holding dough directly in gas combustion flames.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Food Preparation Techniques and Hydrocarbon Formation: Food Chemistry",
            "evidence_type": "educational_content",
            "evidence_summary": "Food safety scientists recommend indirect pan cooking to minimize direct combustion particle accumulation.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 59. protein shakes cancer dumb
    "protein shakes cancer dumb": [
        {
            "claim_subject": "protein powder biological safety profile",
            "claim_text": "Consuming standard dietary protein powder does not cause cancer, brain damage, or intellectual impairment in healthy individuals.",
            "verdict": "TRUE",
            "scientific_truth": "Commercial whey, casein, and plant protein powders are simply isolated food macronutrients. They contain physiological amino acids that are digested normally and do not cause neurodegeneration or carcinogenesis.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Safety Assessment of Whey Protein and Amino Acid Supplementation",
            "evidence_type": "expert_consensus",
            "evidence_summary": "Regulatory food authorities (FDA GRAS, EFSA) confirm the safety of isolated milk and plant protein powders when consumed within normal dietary limits.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "drinking protein shakes destroys brain cells and causes dementia",
            "claim_text": "Drinking gym protein shakes permanently rots your brain cells, lowers your IQ by 50 points, and causes incurable cancer within six months.",
            "verdict": "FALSE",
            "scientific_truth": "This is an irrational fitness smear. Dietary proteins provide essential amino acids that serve as precursors for vital neurotransmitters like dopamine and serotonin without harming cognitive or physical function.",
            "risk_level": "High",
            "medical_domain": "nutrition",
            "population": "adolescents and young adults",
            "source_title": "Nutritional Neuroscience: Amino Acids and Cognitive Function",
            "evidence_type": "educational_content",
            "evidence_summary": "Neurological science confirms amino acids from dietary proteins are essential for normal neuronal structure and neurotransmitter synthesis.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "heavy metal contamination risks in unverified protein supplements",
            "claim_text": "Certain unverified dietary protein brands have been found to contain detectable levels of lead, arsenic, or cadmium due to poor agricultural sourcing and manufacturing.",
            "verdict": "TRUE",
            "scientific_truth": "Independent toxicology screenings (such as Clean Label Project) have identified heavy metal contaminants in low-grade, unregulated protein powders, highlighting the importance of third-party batch testing.",
            "risk_level": "Medium",
            "medical_domain": "public_health",
            "population": "supplement consumers",
            "source_title": "Heavy Metal Contamination in Protein Powders: Consumer Reports",
            "evidence_type": "systematic_review",
            "evidence_summary": "Consumer and toxicology watchdogs recommend purchasing supplements verified by USP, NSF, or Informed-Choice to avoid agricultural heavy metal residues.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 60. cold drinks mango death
    "cold drinks mango death": [
        {
            "claim_subject": "cold drink and mango digestive safety",
            "claim_text": "Drinking a cold carbonated soda after eating mango does not produce deadly poison, cyanide, or carbonic toxic gas in the stomach.",
            "verdict": "TRUE",
            "scientific_truth": "Mangoes contain citric and malic acids, while carbonated beverages contain carbonic acid. Mixing these organic acids in gastric hydrochloric acid (pH 1.5-2.0) produces no toxic chemical reactions or poisonous precipitates.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Gastric Digestive Biochemistry: Guyton and Hall",
            "evidence_type": "educational_content",
            "evidence_summary": "Gastroenterological physiology confirms that mixing fruits with carbonated drinks produces ordinary digestive breakdown without toxic synthesis.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "mango with cold drink creates instant fatal cyanide",
            "claim_text": "Eating mango and immediately drinking a cold soda combines into lethal arsenic and cyanide, causing instant death on the spot.",
            "verdict": "FALSE",
            "scientific_truth": "This is a notorious WhatsApp hoax. Neither fruit pulp nor commercial sodas contain precursors that can biochemically synthesize cyanide or arsenic inside the human stomach.",
            "risk_level": "High",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Fact-Checking Food Combination Myths: FSSAI Clarification",
            "evidence_type": "government_guidance",
            "evidence_summary": "Food Safety and Standards Authority of India (FSSAI) explicitly issued public advisories debunking the mango-cold drink poisoning hoax.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "high sugar and carbonation gastrointestinal bloating",
            "claim_text": "Consuming a large quantity of sugary soda right after sweet mangoes can cause transient abdominal bloating, gas, and digestive discomfort due to high sugar osmotic load.",
            "verdict": "TRUE",
            "scientific_truth": "Ingesting large doses of fructose and sucrose alongside carbonated gas can cause osmotic water shifts and gastric distension, leading to temporary burping and discomfort.",
            "risk_level": "Low",
            "medical_domain": "gastroenterology",
            "population": "general population",
            "source_title": "Fructose Malabsorption and Gastrointestinal Symptoms: Clinical Gastroenterology",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Gastroenterology consensus recognizes benign osmotic bloating from high simultaneous carbohydrate and carbonation intake, distinct from acute toxicity.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 61. cold water blood vessels burst
    "cold water blood vessels burst": [
        {
            "claim_subject": "cold water ingestion hemodynamics",
            "claim_text": "Drinking cold water does not cause internal blood vessels to burst or trigger fatal heart attacks in healthy individuals.",
            "verdict": "TRUE",
            "scientific_truth": "Ingested water rapidly equilibrates with core body temperature as it passes through the esophagus and stomach. While cold liquid can elicit mild transient vagal stimulation, it does not rupture blood vessels.",
            "risk_level": "Low",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Physiological Responses to Cold Water Ingestion: Journal of Applied Physiology",
            "evidence_type": "systematic_review",
            "evidence_summary": "Physiological studies confirm that cold water ingestion does not cause vascular injury or internal hemorrhage.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "cold water after a hot day causes immediate cardiac explosion",
            "claim_text": "Drinking chilled water after being out in the sun causes your heart veins to instantly freeze, shatter, and explode, leading to immediate death.",
            "verdict": "FALSE",
            "scientific_truth": "This is a sensationalist internet forward. The human body has vast thermal capacity and vascular elasticity; cold fluid cannot freeze or shatter coronary vessels.",
            "risk_level": "High",
            "medical_domain": "cardiology",
            "population": "general population",
            "source_title": "Cardiovascular Myths and Facts: American Heart Association",
            "evidence_type": "educational_content",
            "evidence_summary": "AHA confirms cold water hydration during heat is safe and does not cause coronary rupture.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "esophageal cold-induced vasospasm and migraine triggers",
            "claim_text": "Rapid ingestion of ice-cold fluids can stimulate the trigeminal and vagal nerves, causing temporary 'brain freeze' (sphenopalatine ganglion neuralgia) or mild esophageal spasms.",
            "verdict": "PARTLY_TRUE",
            "scientific_truth": "Cold contact with the posterior palate triggers transient rapid constriction and dilation of cerebral meningeal vessels, producing brief, self-limiting cold-stimulus headaches without vascular damage.",
            "risk_level": "Low",
            "medical_domain": "neurology",
            "population": "general population",
            "source_title": "Cold-Stimulus Headache: International Headache Society Criteria",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Neurological classifications confirm ice-induced palatal vasoconstriction triggers transient, benign headaches rather than structural vascular rupture.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 62. curd night cold
    "curd night cold": [
        {
            "claim_subject": "curd consumption at night and respiratory infections",
            "claim_text": "Eating plain yogurt or curd at night does not cause common colds, pneumonia, or respiratory viral infections.",
            "verdict": "TRUE",
            "scientific_truth": "Respiratory infections are caused by airborne viruses (such as rhinoviruses, adenoviruses, or influenza), not by the timing or temperature of dairy consumption. Curd provides beneficial probiotics, calcium, and protein regardless of the hour consumed.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Dairy Intake and Upper Respiratory Tract Infections: Evidence Assessment",
            "evidence_type": "systematic_review",
            "evidence_summary": "Clinical trials find no correlation between yogurt consumption at dinner and increased incidence or severity of viral respiratory tract infections.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating curd after sunset creates instant lethal chest phlegm",
            "claim_text": "Eating curd at night turns your lungs into solid mucus and suffocates you in your sleep.",
            "verdict": "FALSE",
            "scientific_truth": "This is an ancient cultural myth. Dairy proteins do not physically transform into airway mucus; mucosal goblet cell secretion is an immunological and inflammatory response to viral infection, not a byproduct of digested milk curds.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "The Milk-Mucus Myth: A Systematic Review: Archives of Disease in Childhood",
            "evidence_type": "systematic_review",
            "evidence_summary": "Pediatric and respiratory systematic reviews disprove the myth that dairy consumption increases respiratory mucus secretion.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "nocturnal gastroesophageal reflux from heavy late dining",
            "claim_text": "Eating large, cold meals right before lying down can exacerbate gastroesophageal acid reflux and nocturnal coughing in patients with pre-existing GERD.",
            "verdict": "TRUE",
            "scientific_truth": "Lying supine within 2 hours of a heavy meal impairs lower esophageal sphincter competence, allowing gastric acid to irritate the hypopharynx and trigger nocturnal throat clearing, regardless of whether curd was part of the meal.",
            "risk_level": "Low",
            "medical_domain": "gastroenterology",
            "population": "GERD patients",
            "source_title": "ACG Clinical Guideline for the Diagnosis and Management of GERD",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Gastroenterology guidelines recommend avoiding heavy food intake 2-3 hours before bedtime to prevent nocturnal reflux symptoms.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 63. ghee heart damage
    "ghee heart damage": [
        {
            "claim_subject": "ghee saturated fat cardiovascular modulation",
            "claim_text": "Ghee is high in saturated fat and should be consumed in moderation, but incorporating modest amounts into a balanced diet does not inevitably cause heart disease.",
            "verdict": "TRUE",
            "scientific_truth": "Clarified butter (ghee) consists of approximately 60% saturated fatty acids. While excessive intake raises LDL cholesterol, modest culinary use (1-2 teaspoons daily) within overall caloric and saturated fat limits (<10% of energy) is compatible with heart health.",
            "risk_level": "Low",
            "medical_domain": "cardiology",
            "population": "general population",
            "source_title": "Dietary Saturated Fats and Cardiovascular Health: WHO Guidelines",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "WHO and AHA dietary guidelines recommend limiting saturated fats to less than 10% of total daily calories, allowing for modest consumption within a balanced lifestyle.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating half a kilo of ghee daily prevents all heart attacks",
            "claim_text": "Drinking pure warm ghee by the cup lubricates your coronary arteries and guarantees you will never have a heart attack or stroke.",
            "verdict": "FALSE",
            "scientific_truth": "Consuming massive quantities of ghee floods the circulation with saturated palmitic and stearic acids, causing massive elevations in atherogenic ApoB lipoproteins and accelerating coronary atherosclerosis.",
            "risk_level": "High",
            "medical_domain": "cardiology",
            "population": "general population",
            "source_title": "Lipid Profiles and Coronary Risk with High Clarified Butter Ingestion",
            "evidence_type": "systematic_review",
            "evidence_summary": "Cardiovascular epidemiology confirms high dietary intake of clarified butter elevates circulating atherogenic lipids.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "oxidized cholesterol in repeatedly heated ghee",
            "claim_text": "Repeatedly heating ghee to extreme smoking temperatures oxidizes its cholesterol content into oxysterols, which are significantly more atherogenic than fresh ghee.",
            "verdict": "TRUE",
            "scientific_truth": "Thermal degradation of cholesterol during prolonged commercial deep frying generates 7-ketocholesterol and epoxy-cholesterols, which induce vascular endothelial inflammation and cytotoxic plaque progression.",
            "risk_level": "Medium",
            "medical_domain": "cardiology",
            "population": "general population",
            "source_title": "Cholesterol Oxidation Products in Deep-Fried Foods: Food and Chemical Toxicology",
            "evidence_type": "systematic_review",
            "evidence_summary": "Toxicological studies warn that repeatedly reused, overheated cooking fats contain high levels of atherogenic oxysterols.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 64. seed oils poison
    "seed oils poison": [
        {
            "claim_subject": "seed oils linoleic acid cardiovascular safety",
            "claim_text": "Common refined seed oils (such as sunflower, canola, and soybean oil) are not toxic machine lubricants and actually lower cardiovascular disease risk when replacing saturated fats.",
            "verdict": "TRUE",
            "scientific_truth": "Seed oils are rich in polyunsaturated fatty acids (PUFAs), primarily omega-6 linoleic acid. Decades of prospective cohort studies and randomized trials show replacing saturated fats with PUFA-rich vegetable oils lowers serum LDL-C and reduces coronary events.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Circulating Omega-6 Polyunsaturated Fatty Acids and Cardiovascular Disease: Circulation",
            "evidence_type": "systematic_review",
            "evidence_summary": "American Heart Association and Harvard pooled prospective cohort analyses show higher circulating linoleic acid is associated with lower cardiovascular mortality, debunking the 'toxic seed oil' narrative.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "seed oils are industrial toxins that cause all modern diseases",
            "claim_text": "Seed oils are industrial engine degreasers that inflame every cell in your body, cause diabetes, and are deadlier than smoking cigarettes.",
            "verdict": "FALSE",
            "scientific_truth": "This is an unfounded wellness trend. Linoleic acid does not increase systemic inflammatory biomarkers (CRP, IL-6, TNF-alpha) in randomized human trials, and food-grade seed oils undergo strict refining to remove impurities.",
            "risk_level": "High",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Effect of Dietary Linoleic Acid on Markers of Systemic Inflammation: A Systematic Review",
            "evidence_type": "systematic_review",
            "evidence_summary": "Clinical trials demonstrate that increased linoleic acid consumption does not increase systemic inflammatory markers in human subjects.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "repeated deep frying thermal degradation of vegetable oils",
            "claim_text": "Repeatedly reheating vegetable oils to high frying temperatures creates harmful polar compounds, trans fats, and lipid peroxides that should be avoided.",
            "verdict": "TRUE",
            "scientific_truth": "While fresh seed oils are heart-healthy, subjecting polyunsaturated oils to prolonged, repeated commercial deep-frying cycles degrades triglycerides into toxic aldehydes and trans isomers.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Thermal Degradation and Oxidation Products of Reused Vegetable Frying Oils",
            "evidence_type": "expert_consensus",
            "evidence_summary": "Food safety standards worldwide establish strict regulatory thresholds for total polar compounds in commercially reused frying oils.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 65. rain flu myth
    "rain flu myth": [
        {
            "claim_subject": "microbial transmission versus rainwater exposure",
            "claim_text": "Getting soaked in rainwater does not spontaneously generate respiratory viruses; infectious illness requires exposure to an infected person or contaminated aerosols.",
            "verdict": "TRUE",
            "scientific_truth": "Rainwater consists of condensed atmospheric moisture and cannot generate influenza viruses or coronaviruses. Infection strictly requires transmission of viral pathogens between hosts.",
            "risk_level": "Low",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Epidemiology and Prevention of Vaccine-Preventable Diseases: CDC",
            "evidence_type": "educational_content",
            "evidence_summary": "Public health epidemiology confirms viral transmission dynamics are governed by pathogen exposure rather than precipitation.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "one drop of rain on your head creates instant viral pneumonia",
            "claim_text": "If a single drop of rainwater touches your scalp, it immediately transforms into viral pneumonia in your chest.",
            "verdict": "FALSE",
            "scientific_truth": "Water contacting the skin or hair cannot penetrate into pulmonary alveoli or spontaneously create pathogenic bacteria or viruses.",
            "risk_level": "Medium",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Common Cold: Patient Education: UpToDate",
            "evidence_type": "educational_content",
            "evidence_summary": "Medical literature confirms external rain exposure does not induce pulmonary infections in the absence of pathogens.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "hypothermia wet clothes and upper airway vasoconstriction",
            "claim_text": "Remaining in cold, wet clothing for extended periods can cause mild core chilling, which may trigger reflex nasal vasoconstriction and slightly reduce local mucosal defense against pre-existing viral inoculums.",
            "verdict": "PARTLY_TRUE",
            "scientific_truth": "Chilling of the body surface causes reflex sympathetically mediated vasoconstriction in the nasal mucosa, which experimental studies suggest may modestly diminish local immune surveillance if virions are already present.",
            "risk_level": "Low",
            "medical_domain": "infectious_disease",
            "population": "general population",
            "source_title": "Acute Chilling of the Body Surface and Common Cold: Family Practice",
            "evidence_type": "systematic_review",
            "evidence_summary": "Physiological experiments show acute chilling can increase clinical symptom expression in individuals already harboring subclinical viral particles.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 66. cold drinks ebola
    "cold drinks ebola": [
        {
            "claim_subject": "ebola transmission biology and beverage safety",
            "claim_text": "Commercial cold soft drinks and bottled beverages cannot transmit the Ebola virus, which is spread strictly through direct contact with infectious bodily fluids of symptomatic patients.",
            "verdict": "TRUE",
            "scientific_truth": "Ebola is an enveloped filovirus that requires direct exposure to infected blood, vomitus, or secretions. It cannot survive commercial beverage manufacturing, high-acid carbonation (pH 2.5), or hermetic packaging.",
            "risk_level": "Low",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Ebola Virus Disease Transmission: CDC Fact Sheet",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "CDC and WHO confirm that Ebola is not a waterborne or foodborne infection and commercial beverages pose zero transmission risk.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "contaminated cold drinks infected with ebola killed thousands",
            "claim_text": "All mango and cola cold drinks are contaminated with blood from an Ebola patient, and anyone who drinks them dies within 2 hours.",
            "verdict": "FALSE",
            "scientific_truth": "This is an infamous malicious internet hoax originating in 2015. Bottled beverages undergo pasteurization and quality controls, and Ebola cannot spread through commercial retail beverages.",
            "risk_level": "High",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Combating Health Disinformation: WHO Advisory",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "Public health agencies and food safety regulators have repeatedly issued official debunking notices against the cold drink Ebola rumor.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "industrial beverage safety protocols and water sanitization",
            "claim_text": "Commercial beverage bottling plants utilize automated multi-barrier sanitation, UV filtration, and reverse osmosis, preventing microbiological contamination.",
            "verdict": "TRUE",
            "scientific_truth": "Food manufacturing standards (HACCP, ISO 22000) mandate automated multi-stage purification and quality testing, ensuring commercial canned and bottled soft drinks remain pathogen-free.",
            "risk_level": "Low",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Good Manufacturing Practices for Bottled Beverages: FDA",
            "evidence_type": "government_guidance",
            "evidence_summary": "Food safety regulations detail stringent sanitary standards that prevent viral and bacterial adulteration in commercial beverage production.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 67. induction cooking cancer
    "induction cooking cancer": [
        {
            "claim_subject": "induction stove nonionizing magnetic field safety",
            "claim_text": "Cooking food on an induction cooktop does not make food radioactive or increase cancer risk.",
            "verdict": "TRUE",
            "scientific_truth": "Induction cooktops use intermediate-frequency magnetic fields (20-100 kHz) to induce eddy currents in ferromagnetic cookware, generating resistive heat. This non-ionizing radiation leaves no residual energy in food and cannot cause DNA damage.",
            "risk_level": "Low",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Electromagnetic Fields and Induction Hobs: WHO Fact Sheet",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "WHO and International Commission on Non-Ionizing Radiation Protection (ICNIRP) confirm induction cooktop magnetic fields are safe and do not cause cancer.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "induction stoves emit nuclear radiation turning food into poison",
            "claim_text": "Induction stoves emit lethal nuclear gamma radiation that permanently mutates your food's DNA into toxic cancer-causing poisons.",
            "verdict": "FALSE",
            "scientific_truth": "Induction cooktops produce magnetic fields, not ionizing nuclear radiation. The magnetic field drops to near zero a few centimeters from the pan and terminates completely when turned off.",
            "risk_level": "High",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "ICNIRP Guidelines for Limiting Exposure to Electromagnetic Fields",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "ICNIRP standards show stray intermediate magnetic fields from induction cookers are well below international human exposure limits.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "electromagnetic interference precautions for cardiac pacemakers",
            "claim_text": "While induction cooking is cancer-safe, patients with implanted cardiac pacemakers or ICDs should maintain a safe distance (roughly 30-50 cm) to avoid potential electromagnetic interference.",
            "verdict": "TRUE",
            "scientific_truth": "Strong localized electromagnetic fields within close proximity (<30 cm) of an active induction coil can theoretically interact with cardiac rhythm devices, prompting manufacturers to recommend reasonable operational distance.",
            "risk_level": "Medium",
            "medical_domain": "cardiology",
            "population": "pacemaker patients",
            "source_title": "Electromagnetic Interference with Implantable Cardiac Devices: AHA",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Cardiovascular device guidelines suggest maintaining a small distance from active induction surfaces to eliminate electromagnetic sensing artifacts in pacemakers.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 68. onion feet detox
    "onion feet detox": [
        {
            "claim_subject": "invalidation of transdermal plantar detoxification",
            "claim_text": "Placing sliced onions inside your socks does not draw toxins out of internal organs or purify the bloodstream.",
            "verdict": "TRUE",
            "scientific_truth": "The skin of the plantar foot has a thick stratum corneum designed to prevent water loss and pathogen entry; it possesses no anatomical or physiological mechanism to extract systemic metabolic toxins from the circulation.",
            "risk_level": "Low",
            "medical_domain": "alternative_medicine",
            "population": "general population",
            "source_title": "Cutaneous Physiology and Transdermal Drug Delivery: Guyton",
            "evidence_type": "educational_content",
            "evidence_summary": "Dermatological physiology confirms that transdermal toxin extraction via raw vegetables is a biological impossibility.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "raw onion in socks cures the flu and draws out all poisons overnight",
            "claim_text": "Putting raw red onions on the soles of your feet overnight pulls all viruses, bacteria, and heavy metals out of your heart and lungs by morning.",
            "verdict": "FALSE",
            "scientific_truth": "The blackening of onion slices overnight is due to air oxidation of sulfur compounds and enzymes, not extracted bodily toxins. Relying on foot onions for serious infections delays vital antimicrobial care.",
            "risk_level": "High",
            "medical_domain": "alternative_medicine",
            "population": "general population",
            "source_title": "Debunking Folk Remedies: National Center for Complementary and Integrative Health",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "NCCIH clearly debunks onion foot poultices as an ineffective 16th-century superstition lacking empirical validity.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "contact dermatitis and skin maceration from raw onion contact",
            "claim_text": "Leaving raw onions in direct contact with the skin for hours under occlusive socks can cause localized skin irritation, allergic contact dermatitis, and chemical erythema.",
            "verdict": "TRUE",
            "scientific_truth": "Onions contain volatile alkyl sulfoxides and allicin derivatives that can act as primary cutaneous irritants, triggering localized burning, erythema, and maceration when occluded against the skin.",
            "risk_level": "Low",
            "medical_domain": "dermatology",
            "population": "general population",
            "source_title": "Phytodermatitis and Contact Dermatitis from Allium Species: Contact Dermatitis Journal",
            "evidence_type": "systematic_review",
            "evidence_summary": "Dermatology case reports document irritant and allergic contact dermatitis resulting from prolonged folk application of raw allium plants.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 69. roti protein content
    "roti protein content": [
        {
            "claim_subject": "wheat roti carbohydrate and incomplete protein profile",
            "claim_text": "Whole wheat roti contains modest amounts of protein (about 3-4 grams per roti), but it is primarily a carbohydrate staple and is limiting in the essential amino acid lysine.",
            "verdict": "TRUE",
            "scientific_truth": "Whole wheat flour is approximately 70% carbohydrates and 10-12% protein. Cereal proteins (gliadin and glutenin) have low biological value due to lysine deficiency, making roti alone an insufficient sole protein source.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Protein Quality Evaluation: FAO/WHO Expert Consultation",
            "evidence_type": "expert_consensus",
            "evidence_summary": "FAO/WHO nutritional tables classify cereal grains as lysine-deficient, requiring complementary legumes to achieve complete amino acid scores.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating three rotis provides 100 percent of daily protein requirement",
            "claim_text": "Eating three rotis gives you 60 grams of pure muscle-building protein, making lentils, eggs, and dairy completely unnecessary.",
            "verdict": "FALSE",
            "scientific_truth": "Three standard whole wheat rotis provide approximately 9-11 grams of protein, falling far short of the 50-70 grams required daily for adult maintenance and body composition.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Indian Food Composition Tables: ICMR-NIN",
            "evidence_type": "government_guidance",
            "evidence_summary": "Indian Council of Medical Research (ICMR-NIN) food composition tables verify wheat roti provides ~3g protein and over 15g carbohydrates per serving.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "cereal-legume complementation in traditional diets",
            "claim_text": "Combining wheat roti with lysine-rich legumes like dal or chana creates a complementary amino acid profile that significantly improves overall dietary protein quality.",
            "verdict": "TRUE",
            "scientific_truth": "Legumes are high in lysine and low in methionine, whereas cereals are high in methionine and low in lysine; eating them together provides a complete, balanced amino acid profile.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Nutritional Enhancement Through Food Complementation: ICMR",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Nutritional science universally validates the traditional pairing of cereals and pulses to optimize protein bioavailability in plant-based diets.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 70. curd salad banana night dangerous
    "curd salad banana night dangerous": [
        {
            "claim_subject": "nocturnal digestion of curd salad and fruit",
            "claim_text": "Consuming yogurt, fresh salads, or bananas at dinner does not produce toxic compounds or poison the body during sleep.",
            "verdict": "TRUE",
            "scientific_truth": "Human digestive enzymes (pepsin, amylase, lipase) operate continuously day and night. Eating these nutrient-dense whole foods in the evening provides normal calories, potassium, and probiotics without creating metabolic toxins.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Human Digestive Physiology Across the 24-Hour Cycle",
            "evidence_type": "educational_content",
            "evidence_summary": "Gastroenterology textbooks establish that gastrointestinal absorption and enzymatic digestion proceed effectively throughout the sleep cycle.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating banana and curd at night produces deadly toxins",
            "claim_text": "Eating a banana with curd at dinner produces fatal toxins in the gut that cause sudden death before sunrise.",
            "verdict": "FALSE",
            "scientific_truth": "This is an unscientific cultural myth. Curd and bananas are easily digested, supply prebiotics and probiotics, and have never caused toxic poisoning or mortality.",
            "risk_level": "High",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Food Safety and Nutritional Myths: FSSAI",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "National food safety authorities confirm that combining standard healthy food items does not generate toxic chemicals.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "high fiber and nocturnal bloating in sensitive individuals",
            "claim_text": "Eating large raw vegetable salads late at night can cause gas, abdominal distension, and disturbed sleep in individuals prone to irritable bowel syndrome (IBS).",
            "verdict": "PARTLY_TRUE",
            "scientific_truth": "Raw cruciferous vegetables contain high amounts of insoluble fiber and fermentable oligosaccharides (FODMAPs) that produce colonic gas, which can cause discomfort when lying horizontal shortly after eating.",
            "risk_level": "Low",
            "medical_domain": "gastroenterology",
            "population": "IBS patients",
            "source_title": "FODMAPs and Irritable Bowel Syndrome: American College of Gastroenterology",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "ACG guidelines note that high evening raw fiber intake can trigger benign mechanical bloating in visceral hypersensitivity patients.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 71. ghee roti diabetes prevention
    "ghee roti diabetes prevention": [
        {
            "claim_subject": "ghee impact on glycemic index and caloric density",
            "claim_text": "Adding ghee to a roti slightly delays gastric emptying and modestly slows the rate of glucose absorption, but it adds substantial calories and does not cure or prevent diabetes.",
            "verdict": "TRUE",
            "scientific_truth": "Fats slow gastric emptying via cholecystokinin release, which can modestly flatten postprandial glucose curves. However, the total carbohydrate content remains unchanged, and the added saturated fat calories can promote weight gain, worsening insulin resistance.",
            "risk_level": "Low",
            "medical_domain": "endocrinology",
            "population": "diabetics and prediabetics",
            "source_title": "Dietary Fat and Postprandial Glycemia in Type 2 Diabetes: ADA",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "American Diabetes Association notes that adding fat to high-carbohydrate meals delays but often prolongs late postprandial hyperglycemia while adding excess calories.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "smothering roti in ghee makes carbs zero percent sugar",
            "claim_text": "If you cover your roti with thick ghee, the carbohydrates turn into zero sugar, making it 100% safe for diabetics to eat unlimited rotis.",
            "verdict": "FALSE",
            "scientific_truth": "Fat does not erase carbohydrates or stop starch from being enzymatically converted into glucose in the brush border of the small intestine. Overeating ghee-covered rotis leads to severe glycemic spikes and obesity.",
            "risk_level": "High",
            "medical_domain": "endocrinology",
            "population": "diabetics",
            "source_title": "Management of Postprandial Glucose: Endocrine Society",
            "evidence_type": "expert_consensus",
            "evidence_summary": "Endocrinologists emphasize that fat co-ingestion cannot eliminate the glycemic burden of dietary carbohydrates.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "whole grains and fiber for glycemic management",
            "claim_text": "Substituting refined flour with high-fiber whole grain or multigrain rotis containing legumes (like besan) is far more effective for glycemic control than adding large amounts of fat.",
            "verdict": "TRUE",
            "scientific_truth": "Viscous dietary fiber forms a gelatinous matrix in the lumen that physically retards amylase activity and glucose diffusion, safely blunting glycemic excursions without surplus saturated fat calories.",
            "risk_level": "Low",
            "medical_domain": "endocrinology",
            "population": "diabetics",
            "source_title": "Dietary Fiber and Glycemic Control: Systematic Review and Meta-Analysis",
            "evidence_type": "systematic_review",
            "evidence_summary": "Systematic reviews prove fiber enrichment directly lowers postprandial glucose and improves insulin sensitivity in diabetic cohorts.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 72. fish milk white patches
    "fish milk white patches": [
        {
            "claim_subject": "vitiligo autoimmune etiology independent of fish milk pairing",
            "claim_text": "Drinking milk after eating fish does not cause vitiligo, leukoderma, or white patches on the skin.",
            "verdict": "TRUE",
            "scientific_truth": "Vitiligo is an acquired autoimmune condition characterized by CD8+ cytotoxic T-cell destruction of epidermal melanocytes, driven by genetic susceptibility (HLA alleles) and oxidative stress, completely unrelated to dietary combinations.",
            "risk_level": "Low",
            "medical_domain": "dermatology",
            "population": "general population",
            "source_title": "Vitiligo: Pathogenesis and Clinical Practice: British Association of Dermatologists",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Dermatology guidelines conclusively show vitiligo is an autoimmune melanocyte destruction disorder with zero causal link to eating fish and milk together.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "mixing fish and milk triggers incurable instant white skin leprosy",
            "claim_text": "Consuming fish and dairy together creates a violent toxic chemical that instantly destroys skin melanin and causes incurable skin leprosy.",
            "verdict": "FALSE",
            "scientific_truth": "This is a widespread cultural misconception without biological plausibility. Coastal communities worldwide consume fish chowders and creamy seafood curries daily without elevated rates of vitiligo.",
            "risk_level": "Medium",
            "medical_domain": "dermatology",
            "population": "general population",
            "source_title": "Debunking Cultural Dermatology Myths: Indian Journal of Dermatology",
            "evidence_type": "expert_consensus",
            "evidence_summary": "Indian dermatological reviews explicitly debunk the fish-and-milk myth, noting millions consume seafood with dairy without skin depigmentation.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "seafood allergy versus autoimmune skin depigmentation",
            "claim_text": "Some individuals may experience IgE-mediated allergic reactions to seafood (such as hives, angioedema, or anaphylaxis), which are distinct from autoimmune vitiligo.",
            "verdict": "TRUE",
            "scientific_truth": "Fish allergies involve parvalbumin sensitization triggering immediate mast cell degranulation and urticaria, a distinct acute immunological pathway with no connection to permanent melanocyte loss.",
            "risk_level": "Low",
            "medical_domain": "immunology",
            "population": "allergic individuals",
            "source_title": "Seafood Allergy: Mechanisms and Management: WAO Guidelines",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "World Allergy Organization clarifies that food allergies present with urticaria and anaphylaxis, never autoimmune depigmentation.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 73. turmeric milk cure all
    "turmeric milk cure all": [
        {
            "claim_subject": "turmeric milk supportive comfort versus medical replacement",
            "claim_text": "Turmeric milk (haldi doodh) provides mild anti-inflammatory comfort and warm hydration, but it cannot cure major medical conditions like bone fractures, diabetes, or cancer.",
            "verdict": "TRUE",
            "scientific_truth": "While curcumin has mild in vitro anti-inflammatory properties and milk provides protein and calcium, systemic curcumin absorption is extremely low; it cannot set displaced bone fractures or eliminate systemic malignancies.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Curcumin: A Review of Its Effects on Human Health: Foods",
            "evidence_type": "systematic_review",
            "evidence_summary": "Nutritional reviews establish that dietary turmeric provides modest supportive antioxidant benefits but lacks clinical potency to substitute for medical therapeutics.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "haldi doodh heals broken bones and cures cancer in 3 days",
            "claim_text": "Drinking a glass of turmeric milk cures compound bone fractures in 3 days and eliminates all stage 4 cancer without hospital treatment.",
            "verdict": "FALSE",
            "scientific_truth": "Relying on turmeric milk for untreated fractures leads to malunion, nonunion, permanent deformity, and limb dysfunction, while delaying cancer care is fatal.",
            "risk_level": "High",
            "medical_domain": "emergency_medicine",
            "population": "trauma patients",
            "source_title": "Principles of Fracture Management: AO Foundation",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Orthopedic surgery guidelines emphasize rigid anatomic reduction and immobilization for skeletal fractures, which dietary drinks cannot achieve.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "dietary fat enhancement of curcumin absorption",
            "claim_text": "Consuming turmeric with dietary fats (such as warm milk containing milkfat) modestly improves the intestinal absorption of lipophilic curcumin compared to water.",
            "verdict": "TRUE",
            "scientific_truth": "Curcumin is highly hydrophobic; incorporation into mixed lipid micelles during dairy fat digestion modestly enhances enterocyte uptake compared to aqueous suspensions.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Lipid-Based Delivery Systems for Enhancing Curcumin Bioavailability",
            "evidence_type": "systematic_review",
            "evidence_summary": "Pharmacokinetic studies demonstrate lipid vehicles improve intestinal solubility of dietary curcumin compounds.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 74. honey hot water toxic
    "honey hot water toxic": [
        {
            "claim_subject": "honey in warm beverages safety and hmf formation",
            "claim_text": "Adding honey to warm tea or hot water does not create deadly poisons, toxins, or carcinogens.",
            "verdict": "TRUE",
            "scientific_truth": "Warming honey dissolves sugars and can modestly increase hydroxymethylfurfural (HMF), a natural Maillard byproduct found in all baked goods, toasted bread, and coffee, at levels that are completely non-toxic to humans.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Hydroxymethylfurfural (HMF) in Foods: Toxicological and Food Safety Evaluation",
            "evidence_type": "systematic_review",
            "evidence_summary": "Comprehensive toxicological reviews by EFSA confirm dietary levels of HMF formed during culinary heating pose no human health hazard.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "heated honey turns into instant deadly poison",
            "claim_text": "Heating honey above 40 degrees Celsius transforms it into a deadly toxic venom that destroys internal organs.",
            "verdict": "FALSE",
            "scientific_truth": "This is an erroneous interpretation of ancient Ayurvedic texts. Commercial honey is routinely pasteurized at 60-70C worldwide for shelf stability and consumed by billions without toxic harm.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Pasteurization and Quality Standards of Commercial Honey: Codex Alimentarius",
            "evidence_type": "expert_consensus",
            "evidence_summary": "Codex Alimentarius and international food standards confirm pasteurized heated honey meets all human safety criteria.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "heat degradation of honey enzymes and aromatics",
            "claim_text": "While heating honey is completely safe, boiling it can denature delicate beneficial enzymes (like diastase and glucose oxidase) and volatilize floral aroma compounds.",
            "verdict": "TRUE",
            "scientific_truth": "High thermal exposure degrades endogenous honey enzymes and reduces antioxidant phenolic capacity, diminishing its raw nutritional quality without producing toxic substances.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Effect of Thermal Treatment on Honey Quality: Journal of Food Science",
            "evidence_type": "systematic_review",
            "evidence_summary": "Food chemistry research shows heat alters enzyme activity and flavor profiles while confirming food safety is fully maintained.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 75. milk fish digestion incompatible
    "milk fish digestion incompatible": [
        {
            "claim_subject": "physiological compatibility of fish and dairy proteins",
            "claim_text": "Consuming fish and milk together is physiologically safe; both foods are broken down normally by gastric pepsin and pancreatic proteases without incompatibility.",
            "verdict": "TRUE",
            "scientific_truth": "Proteins from aquatic and mammalian sources consist of standard peptide bonds. The human gastrointestinal tract hydrolyzes all dietary proteins into tripeptides, dipeptides, and free amino acids regardless of food combination.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Protein Digestion and Absorption: Human Physiology",
            "evidence_type": "educational_content",
            "evidence_summary": "Gastroenterological science confirms that digestive proteolytic enzymes handle diverse animal proteins simultaneously without adverse interaction.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating fish and milk together creates explosive stomach poison",
            "claim_text": "Combining fish and milk in the stomach creates an explosive poison that instantly rots the intestines.",
            "verdict": "FALSE",
            "scientific_truth": "Traditional culinary classics worldwide—such as British fish pie, French salmon in cream sauce, and clam chowder—combine seafood and dairy safely every day.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Global Culinary Nutrition and Food Safety",
            "evidence_type": "educational_content",
            "evidence_summary": "Global food safety and cross-cultural nutritional data prove seafood and dairy co-consumption is completely safe.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "scombroid poisoning from improperly stored fish",
            "claim_text": "If fish is improperly stored and develops high histamine levels, eating it can cause scombroid food poisoning, which can be mistakenly blamed on dairy consumed at the same meal.",
            "verdict": "TRUE",
            "scientific_truth": "Bacterial histidine decarboxylation in poorly refrigerated scombroid fish generates high histamine levels, producing flushing, headache, and diarrhea that is entirely due to fish spoilage, not milk interaction.",
            "risk_level": "Low",
            "medical_domain": "infectious_disease",
            "population": "general population",
            "source_title": "Scombroid Fish Poisoning: FDA Bad Bug Book",
            "evidence_type": "government_guidance",
            "evidence_summary": "FDA foodborne illness guidance clarifies scombroid intoxication is caused by bacterial histamine accumulation in spoiled fish.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 76. citrus milk dangerous
    "citrus milk dangerous": [
        {
            "claim_subject": "gastric acid curdling of milk is normal physiology",
            "claim_text": "Consuming citrus fruit and milk together is harmless because milk is naturally curdled by strong hydrochloric acid in the stomach as the first step of normal digestion.",
            "verdict": "TRUE",
            "scientific_truth": "Stomach acid has a pH of 1.5-2.0, far more acidic than lemon juice (pH ~2.5). Ingested milk always immediately precipitates into casein curds upon hitting gastric acid; this is essential for pepsin to hydrolyze milk proteins.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Physiology of the Gastrointestinal Tract: Johnson",
            "evidence_type": "educational_content",
            "evidence_summary": "Physiological science establishes that milk precipitation is an obligatory, healthy stage of normal gastric digestion.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "drinking lemon juice with milk creates toxic stomach stones",
            "claim_text": "Drinking orange juice or lemon with milk creates toxic stone-like curds in the stomach that cause fatal bowel gangrene.",
            "verdict": "FALSE",
            "scientific_truth": "Acid-coagulated casein forms soft, digestible curds (just like paneer, cottage cheese, or ricotta), which are readily broken down by gastric pepsin and pancreatic enzymes.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Food Chemistry: Casein Micelle Coagulation Mechanisms",
            "evidence_type": "educational_content",
            "evidence_summary": "Dairy science demonstrates that acid-induced casein curdling is identical to cheese-making and produces completely digestible protein matrices.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "acidic curdling culinary texture versus physiological tolerance",
            "claim_text": "While mixing citrus juice directly into a glass of warm milk curdles the liquid unpleasantly, it poses no danger to digestion and is the basis of desserts like lemon posset.",
            "verdict": "TRUE",
            "scientific_truth": "Direct mixing lowers milk pH below the casein isoelectric point (pH 4.6), causing curd separation; while culinary texture may be unappealing unless intended, it is entirely non-toxic.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Principles of Food Science: Acid-Base Food Chemistry",
            "evidence_type": "educational_content",
            "evidence_summary": "Culinary food chemistry explains that milk curdling by fruit acids is a physical emulsion change with zero toxicological risk.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 77. curd mango toxins
    "curd mango toxins": [
        {
            "claim_subject": "mango yogurt nutritional safety and digestion",
            "claim_text": "Combining fresh mango with yogurt or curd is completely safe, highly nutritious, and does not create internal toxins or metabolic damage.",
            "verdict": "TRUE",
            "scientific_truth": "Mangoes provide carotenoids, vitamin C, and fiber, while curd provides probiotics, calcium, and protein. They are digested synergistically without producing pathogenic metabolites or endotoxins.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Nutritional Value of Mango and Fermented Dairy Combinations",
            "evidence_type": "educational_content",
            "evidence_summary": "Nutritional science documents that combining fruit carbohydrates with dairy protein blunts postprandial glycemic response safely.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating mango and curd together causes severe blood toxicity",
            "claim_text": "Eating mango and curd together clashes opposing bodily energies and poisons your blood, causing skin boils and organ failure.",
            "verdict": "FALSE",
            "scientific_truth": "This is a baseless folk superstition. Popular foods worldwide—such as mango lassi, mango shrikhand, and mango parfaits—are consumed safely by millions without toxic effects.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Dietary Myths and Realities: National Institute of Nutrition",
            "evidence_type": "government_guidance",
            "evidence_summary": "National Institute of Nutrition (NIN) refutes fruit-and-curd toxicity myths, affirming their nutritional suitability.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "caloric and carbohydrate density of sweetened mango desserts",
            "claim_text": "While mango and curd are healthy, consuming sweetened mango lassi with heavy added sugar can significantly elevate caloric and glycemic intake in diabetics.",
            "verdict": "TRUE",
            "scientific_truth": "Commercial fruit yogurts and traditional lassis often contain high amounts of added refined sugar, which can trigger acute glycemic spikes in individuals with impaired glucose tolerance.",
            "risk_level": "Low",
            "medical_domain": "endocrinology",
            "population": "diabetics",
            "source_title": "Glycemic Management and Added Sugars in Dairy Beverages: ADA",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Diabetes guidelines advise monitoring added sugar content in processed dairy and fruit preparations to maintain glycemic control.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 78. water after fruit harmful
    "water after fruit harmful": [
        {
            "claim_subject": "water intake after fruit consumption safety",
            "claim_text": "Drinking water after eating fruit (such as watermelon or apples) does not dilute digestive acids to cause cholera, intestinal rot, or severe illness.",
            "verdict": "TRUE",
            "scientific_truth": "Fruits like watermelon and oranges are already over 85-92% water. Ingesting additional water does not overwhelm gastric digestive capacity or inactivate digestive enzymes, which adaptively adjust secretagogue output.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Human Gastric Secretory Dynamics: Textbook of Gastroenterology",
            "evidence_type": "educational_content",
            "evidence_summary": "Gastroenterological research confirms the stomach regulates gastric juice secretion dynamically, debunking claims that drinking water dilutes enzymes dangerously.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "drinking water after watermelon causes instant cholera and death",
            "claim_text": "Drinking water after eating watermelon causes your intestines to instantly burst with cholera bacteria, killing you in minutes.",
            "verdict": "FALSE",
            "scientific_truth": "Cholera is caused exclusively by the bacterium Vibrio cholerae acquired through fecally contaminated water or food, not by drinking clean water after eating fruit.",
            "risk_level": "High",
            "medical_domain": "infectious_disease",
            "population": "general population",
            "source_title": "Vibrio cholerae Transmission and Pathogenesis: WHO",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "WHO confirms cholera etiology requires ingestion of the specific bacterial pathogen Vibrio cholerae.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "fructose malabsorption and rapid fluid transit in sensitive guts",
            "claim_text": "Drinking excessive volumes of water immediately after eating large amounts of high-fructose fruits can cause loose stools or abdominal cramps in individuals with fructose malabsorption.",
            "verdict": "PARTLY_TRUE",
            "scientific_truth": "In individuals with limited GLUT5 fructose transport capacity, unabsorbed fructose exerts an osmotic laxative effect in the colon, which can be accentuated by rapid fluid transit.",
            "risk_level": "Low",
            "medical_domain": "gastroenterology",
            "population": "individuals with fructose malabsorption",
            "source_title": "Pathophysiology of Dietary Fructose-Induced Diarrhea",
            "evidence_type": "systematic_review",
            "evidence_summary": "Gastroenterological reviews describe osmotic diarrhea from excess unabsorbed fructose, distinct from infectious disease or toxicity.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 79. mango milk dangerous
    "mango milk dangerous": [
        {
            "claim_subject": "mango milkshake nutritional compatibility",
            "claim_text": "Drinking a mango milkshake is safe and wholesome for healthy individuals, providing vitamin A, potassium, calcium, and protein.",
            "verdict": "TRUE",
            "scientific_truth": "Mango flesh contains carbohydrates and organic acids that do not cross-react deleteriously with dairy caseins or whey; enzymatic digestion in the human gut readily hydrolyzes both components.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Nutrient Synergy in Fruit-Dairy Blends: International Dairy Journal",
            "evidence_type": "educational_content",
            "evidence_summary": "Food nutritional science confirms combining mango pulp with cow's milk produces a stable, digestible beverage without creating toxins.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "mango milkshake creates toxic sludge that destroys the liver",
            "claim_text": "Mango milkshake forms an indestructible toxic sludge in the stomach that permanently clogs liver bile ducts.",
            "verdict": "FALSE",
            "scientific_truth": "This is an unfounded wellness fear. Liquid food blends are processed normally into chyme and pass through the pylorus within 1-2 hours without forming sludge in the biliary system.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Gastric Emptying and Chyme Formation: Physiology Reviews",
            "evidence_type": "educational_content",
            "evidence_summary": "Gastrointestinal physiology confirms chyme liquefaction occurs rapidly and completely for fruit-dairy suspensions.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "lactose intolerance symptoms mistaken for fruit-dairy toxicity",
            "claim_text": "Individuals with lactose intolerance may experience cramps, gas, or diarrhea after drinking a mango shake made with dairy milk, symptoms due to lactase deficiency rather than fruit-dairy incompatibility.",
            "verdict": "TRUE",
            "scientific_truth": "Undigested lactose undergoes bacterial fermentation in the colon when brush-border lactase is deficient; switching to lactose-free milk or plant milk eliminates symptoms completely.",
            "risk_level": "Low",
            "medical_domain": "gastroenterology",
            "population": "lactose-intolerant individuals",
            "source_title": "Lactose Intolerance: Clinical Diagnosis and Management: UpToDate",
            "evidence_type": "clinical_guideline",
            "evidence_summary": "Clinical gastroenterology clarifies that post-dairy gastrointestinal symptoms stem from lactase enzyme deficiency rather than food incompatibility.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 80. rice dinner weight gain
    "rice dinner weight gain": [
        {
            "claim_subject": "white rice at dinner and thermodynamic weight balance",
            "claim_text": "Eating white rice at dinner does not cause weight gain if total daily calorie intake remains within your body's energy requirements.",
            "verdict": "TRUE",
            "scientific_truth": "Adipose tissue accumulation is determined by net 24-hour thermodynamic energy balance; carbohydrates consumed at night are metabolized or stored as glycogen identically to daytime consumption if caloric balance is maintained.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Dietary Macronutrients, Meal Timing, and Body Composition: Systematic Review",
            "evidence_type": "systematic_review",
            "evidence_summary": "Randomized controlled trials comparing nighttime vs daytime carbohydrate intake show no difference in weight loss when total calories and protein are matched.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating rice after 7pm turns into 100 percent pure fat overnight",
            "claim_text": "If you eat even a single grain of rice after 7 PM, your body automatically shuts down metabolism and turns 100% of it into visceral belly fat while you sleep.",
            "verdict": "FALSE",
            "scientific_truth": "Basal metabolic rate continues during sleep, consuming significant energy for cardiac output, ventilation, and cellular protein turnover; night-time carbohydrate intake does not bypass thermodynamics.",
            "risk_level": "Medium",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Nocturnal Energy Expenditure and Metabolic Rate During Sleep",
            "evidence_type": "educational_content",
            "evidence_summary": "Metabolic studies show nighttime basal energy expenditure remains substantial, invalidating the claim that evening carbs bypass metabolic utilization.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "portion control and glycemic load of white rice",
            "claim_text": "Because white rice has a relatively high glycemic index and lower satiety per volume than whole grains, practicing portion control and pairing it with vegetables and protein is beneficial for weight management.",
            "verdict": "TRUE",
            "scientific_truth": "Refined polished white rice has lower fiber content, which can facilitate passive overconsumption; pairing with protein and soluble fiber lowers the effective glycemic index and enhances satiety.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Glycemic Index of Rice Varieties and Impact on Satiety",
            "evidence_type": "systematic_review",
            "evidence_summary": "Dietary guidelines advise portion moderation and pairing refined grains with fibrous vegetables and lean proteins to optimize satiety.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 81. sugar feeds cancer
    "sugar feeds cancer": [
        {
            "claim_subject": "systemic glucose homeostasis and tumor metabolism",
            "claim_text": "All human cells rely on glucose for basal energy, and dietary sugar restriction does not selectively starve cancer cells while sparing healthy tissue.",
            "verdict": "TRUE",
            "scientific_truth": "While neoplastic cells utilize high rates of aerobic glycolysis (Warburg effect), systemic blood glucose is strictly maintained via hepatic glycogenolysis and gluconeogenesis; tumors adapt to consume amino acids and lactate if glucose is restricted.",
            "risk_level": "Low",
            "medical_domain": "oncology",
            "population": "cancer patients",
            "source_title": "Cancer Metabolism and Dietary Interventions: Nature Reviews Cancer",
            "evidence_type": "systematic_review",
            "evidence_summary": "Oncology consensus confirms that tumors extract metabolic substrates from the host even during severe dietary carbohydrate restriction.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "eating a teaspoon of sugar accelerates cancer growth 100 times",
            "claim_text": "Eating any sugar immediately pours fuel directly on cancer tumors, multiplying their growth rate by 100 times in 24 hours.",
            "verdict": "FALSE",
            "scientific_truth": "Cancer progression is governed by oncogenic mutations, tumor microenvironments, and angiogenesis, not acute postprandial glycemic spikes. Patients starving themselves of carbohydrates risk severe cachexia and reduced tolerance to chemotherapy.",
            "risk_level": "High",
            "medical_domain": "oncology",
            "population": "cancer patients",
            "source_title": "Dietary Myths and Cancer: American Institute for Cancer Research",
            "evidence_type": "expert_consensus",
            "evidence_summary": "AICR explicitly debunks the myth that sugar selectively feeds tumors, cautioning that severe self-imposed dietary restrictions cause malnutrition.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "chronic sugar overconsumption and hyperinsulinemia in cancer risk",
            "claim_text": "Chronic high intake of added sugars can promote visceral adiposity, hyperinsulinemia, and elevated IGF-1 signaling, which are established risk factors for several cancers.",
            "verdict": "TRUE",
            "scientific_truth": "Excess caloric intake from sugar-sweetened beverages fosters obesity, which elevates circulating insulin, IGF-1, and inflammatory cytokines, creating an environment favorable to tumorigenesis.",
            "risk_level": "Low",
            "medical_domain": "oncology",
            "population": "general population",
            "source_title": "Insulin, IGF-1, and Cancer Risk: Endocrine Reviews",
            "evidence_type": "systematic_review",
            "evidence_summary": "Epidemiological and mechanistic studies confirm indirect links between chronic hyperinsulinemia/obesity and increased cancer incidence.",
            "twist_direction": "nuanced_qualification"
        }
    ],

    # 82. microwave cancer
    "microwave cancer": [
        {
            "claim_subject": "microwave nonionizing radiation food safety",
            "claim_text": "Microwave ovens heat food using non-ionizing electromagnetic radiation, which does not make food radioactive, alter DNA, or cause cancer.",
            "verdict": "TRUE",
            "scientific_truth": "Microwaves operate at 2.45 GHz, exciting polar water molecules through dielectric heating. This non-ionizing radiation lacks the quantum energy needed to break chemical bonds, ionize atoms, or induce carcinogenesis.",
            "risk_level": "Low",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "Microwave Ovens and Food Safety: WHO Fact Sheet",
            "evidence_type": "public_health_guidance",
            "evidence_summary": "World Health Organization and FDA confirm food cooked in microwave ovens is completely safe and free from radiation or carcinogens.",
            "twist_direction": "factual_inversion"
        },
        {
            "claim_subject": "microwaved food is radioactive poison that causes terminal leukemia",
            "claim_text": "Microwaving your food completely destroys all molecular structure and turns every meal into lethal radioactive poison that gives you leukemia.",
            "verdict": "FALSE",
            "scientific_truth": "Microwaving simply generates heat via molecular vibration. The radiation is completely contained within the Faraday cage of the oven and terminates the millisecond the door opens or the timer stops.",
            "risk_level": "High",
            "medical_domain": "public_health",
            "population": "general population",
            "source_title": "FDA: Microwave Oven Radiation Safety",
            "evidence_type": "government_guidance",
            "evidence_summary": "FDA enforces strict emission standards ensuring microwave ovens leak negligible non-ionizing energy and impart zero radiation to food.",
            "twist_direction": "viral_hyperbole"
        },
        {
            "claim_subject": "nutrient retention in microwave cooking versus boiling",
            "claim_text": "Because microwave cooking requires shorter heating times and minimal added water, it often preserves heat-sensitive vitamins (like vitamin C) better than prolonged boiling.",
            "verdict": "TRUE",
            "scientific_truth": "Water-soluble vitamins (C and B complex) leach into boiling water; microwave steaming minimizes thermal exposure time and water volume, yielding superior retention of bioactive nutrients.",
            "risk_level": "Low",
            "medical_domain": "nutrition",
            "population": "general population",
            "source_title": "Effects of Cooking Methods on Nutrient Retention: Journal of Food Quality",
            "evidence_type": "systematic_review",
            "evidence_summary": "Nutritional food science confirms microwave cooking retains higher concentrations of water-soluble vitamins compared to conventional boiling.",
            "twist_direction": "nuanced_qualification"
        }
    ]
}
