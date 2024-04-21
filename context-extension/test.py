import SelfExtend
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig
import torch
import torch.nn as nn
import numpy as np
import vllm

#model = AutoModelForCausalLM.from_pretrained("/pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000", local_files_only=True)
# llama 7b tokenzier
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

input_text = """
Name:  ___                     Unit No:   ___
 
Admission Date:  ___              Discharge Date:   ___
 
Date of Birth:  ___             Sex:   M
 
Service: MEDICINE
 
Allergies: 
Codeine / Percocet / Sulfa (Sulfonamide Antibiotics)
 
Attending: ___
 
Chief Complaint:
increaswed pulmonary secretions and fevers
 
Major Surgical or Invasive Procedure:
___
Bronchoscopy

___
Bronchoscopy
EGD

___
Bronchoscopy

___
Laryngoscopy

___
Tracheostomy tube replaced

___
Tracheostomy tube replaced

 
History of Present Illness:
___ w/h/o ESRD s/p kidney transplant in ___ & ___, pancreas
transplant in ___ TBM s/p right thoractomy, 
tracheobronchoplasty
in ___ with airway occlusion s/p emergent cric and open
trach, with subsequent hospitalization stay complicated by
gastric ulcer perforation s/p ex. lap, primary gastric repair,
___ patch ___, as well as embolic strokes to thoracic spinal
cord and brain causing BLE paresis.  He was then re-hospitalized
with worsening mixed respiratory/metabolic acidosis requiring
emergent HD, and his course was complicated by respiratory
difficulty ultimately requiring debridement of suprastomal
granulation tissue, and replacement ___ trach button. 
He now returns to the ED from ___ rehab with persistent
nausea/vomiting since yesterday afternoon, with concern by rehab
staff for aspiration event, now with increasing lethargy,
increased bronchial sounds, and putative new infiltrate on CXR 
at
rehab.

 
Past Medical History:
# Diabetes mellitus type I, now Diabetes mellitus type II post  
pancreas transplant (failed)  
# Status post renal (___), pancreas transplants (___), kidney  

transplant ___  
# Tracheobronchomalacia, severe. medical optimization since 
___  
# CKD Baseline Cr 1.1-1.5 this year  
# Hypertension  
# GERD  
# HLD  
# Peptic ulcer disease  
# ___ esophagitis  
# Right lower extremity cellulitis  
# Left fifth toe amputation for Gangrene  
# Charcot Arthropathy- Septic left subtalar joint  
# Urinary tract infections  
# Retinopathy, status post vitrectomy  
# Esophageal achalasia  
# Post-strep GN  
# h/o stage 1 colon ca s/p resection in ___  
# s/p venous graft surgery  
# Right IJ thrombus ___

 
Social History:
___
Family History:
No lung cancer or congenital lung disease.
Mother had frequent bronchitis  

 
Physical Exam:
97.4 99 129/68 22 98% 4L N/C 
Gen: Uncomfortable, lethargic
CV: RRR, no M/R/G
Pulm: rhonchorous bilaterally
___: distended, mod epigastric tenderness
Ext: 1+ edema
GU: foley in place draining generally clear urine
Skin: CVL site without purulence/erythema, R thoracostomy site
w/o e/o infection
.
Discharge Exam:
Patient in bed 99% on 40% TM
Trach in place
Clear lungs on auscultation with no secretions at his trach site
RRR no m/r/g
Abd soft, non-tender, non-distended
warm extremities with RLE ulcer pink borders and white film and 
pink granulation on the wound
Coccyx ulcer with some dark boggy tissue
Unable to move lower extremitiies
Peripheral IV in place, Dobhoff in nostril
 
Pertinent Results:
CBC:
___ 11:30AM BLOOD WBC-13.4*# RBC-3.38* Hgb-9.9* Hct-31.5* 
MCV-93 MCH-29.3 MCHC-31.4 RDW-16.0* Plt ___
___ 05:16AM BLOOD WBC-9.4 RBC-2.73* Hgb-7.9* Hct-26.2* 
MCV-96 MCH-28.9 MCHC-30.1* RDW-16.4* Plt ___
___:53AM BLOOD WBC-8.3 RBC-2.97* Hgb-8.7* Hct-28.2* 
MCV-95 MCH-29.3 MCHC-30.9* RDW-16.0* Plt ___
___ 07:00AM BLOOD WBC-13.0* RBC-2.67* Hgb-7.9* Hct-26.4* 
MCV-99* MCH-29.6 MCHC-29.9* RDW-17.4* Plt ___

COAGS:
___ 05:32AM BLOOD ___ PTT-38.2* ___
___ 10:42AM BLOOD ___ PTT-61.4* ___
___ 04:45PM BLOOD ___ PTT-50.3* ___
___ 07:00AM BLOOD ___ PTT-48.6* ___
___ 06:50AM BLOOD ___ PTT-37.4* ___

BMP:
___ 11:30AM BLOOD Glucose-236* UreaN-45* Creat-2.2* Na-139 
K-5.3* Cl-107 HCO3-19* AnGap-18
___ 05:16AM BLOOD Glucose-153* UreaN-51* Creat-2.6* Na-143 
K-4.0 Cl-110* HCO3-16* AnGap-21*
___ 01:58AM BLOOD Glucose-151* UreaN-37* Creat-2.1* Na-138 
K-4.0 Cl-109* HCO3-23 AnGap-10
___ 03:06AM BLOOD Glucose-167* UreaN-45* Creat-1.9* Na-141 
K-4.7 Cl-115* HCO3-17* AnGap-14
___ 04:24AM BLOOD Glucose-347* UreaN-83* Creat-1.4* Na-133 
K-5.7* Cl-100 HCO3-29 AnGap-10
___ 07:00AM BLOOD Glucose-273* UreaN-80* Creat-1.2 Na-139 
K-5.4* Cl-110* HCO3-21* AnGap-13

LFT:
___ 11:30AM BLOOD ALT-13 AST-13 AlkPhos-87 TotBili-0.1
___ 06:30AM BLOOD ALT-10 AST-8 AlkPhos-86 TotBili-0.1

ELECTROLYTES:
___ 06:06PM BLOOD Calcium-10.3 Phos-4.5 Mg-2.0
___ 02:45AM BLOOD Calcium-9.0 Phos-3.2 Mg-1.8
___ 07:00AM BLOOD Calcium-10.0 Phos-3.3 Mg-2.5

OTHER:
___ 03:48AM BLOOD calTIBC-131* Ferritn-344 TRF-101*
___ 07:00AM BLOOD VitB12-575 Folate-18.7
___ 04:40AM BLOOD PTH-74*
___ 05:33AM BLOOD 25VitD-17*
___ 04:00PM BLOOD Cortsol-12.5

TACRO:
___ 11:56AM BLOOD tacroFK-5.1
___ 06:50AM BLOOD tacroFK-8.2

___ CT Chest/Abd/Pelvis :
1.  Improved but mildly persistent chronic phlegmon with few 
residual foci of intrinsic air noted extending from the inferior 
stomach antrum towards the anterior abdominal wall with 
decreased amount of intrinsic, but extraperitoneal air in this 
phlegmonous change. 
2.  Nonspecific new foci of air within the hilum of the right 
pelvic kidney transplant which could be iatrogenic from foley 
catheterization, however, underlying infection cannot be 
excluded. Recommend close follow up of this. 
3.  New central left upper lobe and right lower lobe airspace 
consolidation with near complete resolution of the left lower 
lobe airspace consolidation.  Probable loculated right pleural 
effusion.  
4.  Stable thickening of the tracheobronchial tree with midline 
tracheostomy. 
  
___ EGD :

Moderate erythema and scattered ulceration in mid esophagus.
Retained fluids in stomach
Scattered erosions in body of the stomach consistent with NG 
tube trauma.
Small mucosal defect in D1 at site of surgical anastomosis. 
Remainder of duodenal mucosa normal.
Otherwise normal EGD to third part of the duodenum

___ CXR :

As compared to the previous radiograph, the tracheal button has 
been replaced.  The lungs appear better ventilated than at the 
previous 
examination and the pre-existing signs suggestive of 
mild-to-moderate 
pulmonary edema have decreased in severity in the interval.  The 
retrocardiac lung areas are better ventilated than on the 
previous examination.  There also is better ventilation of the 
right lung base.  However, small bilateral pleural effusions 
persist.  In the well ventilated areas of the lung parenchyma, 
there is no evidence of new parenchymal opacities. 

___ Post pyloric FT placement :

Successful repositioning of Dobbhoff tube into the post-pyloric 
position.  The tube is ready to use. 

___ RENAL U/S:
FINDINGS:  The renal morphology is normal.  Specifically, the 
cortex is normal in thickness and echogenicity, the pyramids and 
renal sinus are normal.  There is no pelvi-infundibular 
thickening.  There is no hydronephrosis or perinephric fluid 
collection. 
  
The resistive indices of the intrarenal arteries range from 
0.75-0.77, mildly elevated but improved, previously ranging from 
0.85-0.87.  The acceleration times and peak systolic velocity of 
the main renal artery is normal, measuring 32.9 cm/sec and 
previously measuring 37.6 cm/sec.  The vascularity is asymmetric 
throughout the transplant.  The renal vein is patent and shows a 
normal waveform. 

___ CT ABD/PELVIS:
IMPRESSION: 
1.  Known prior gastric perforation, with improved right rectus 
sheath 
collection, but persistent sinus with mild enlargement of a 
small gas and 
fluid-containing collection anteroineferior to the pylorus. No 
evidence for active leak. 
2.  Mild dilation of the proximal small bowel loops measuring up 
to 2.9 cm with decompression of the distal loops without focal 
transition; a partial small-bowel obstruction is a condieration 
though a single site can not be identified. 
3.  Extensive abdominal aortic atherosclerosis. 
4.  Moderate-sized right pleural effusion with compressive right 
basilar 
atelectasis, slightly worse.   

___ CT CHEST:
IMPRESSION: 
1.  Right lower lobe dependent opacity associated with bronchial 
occlusion is likely due to atelectasis secondary to mucus 
plugging.  Coexisting infection cannot be excluded.  
2.  Small right pleural effusion. 
3.  Mildly enlarged heterogeneous thyroid gland is incompletely 
imaged.  A thyroid ultrasound may be considered for further 
characterization if one has recently not been performed, if 
clinically indicated.   
4.  Pulmonary arterial enlargement likely secondary to pulmonary 
arterial 
hypertension. 

___ CT ABD/PELVIS:
IMPRESSION: 
1.  Unchanged right rectus sheath air collection with suspected 
antral/antropyloric fistula with associated ill-defined area of 
soft tissue in the adjacent transverse mesocolon. 
2.  Minimally thickened gallbladder wall and possible small 
gallstones are not diagnostic for acute gallbladder disease, 
however if clinical suspicion exists, a right upper quadrant 
ultrasound would be helpful for further evaluation.   
3.  Extensive abdominal aortic atherosclerosis. 
4.  Unchanged right basilar atelectasis with small right pleural 
effusion. 

___ MRI L SPINE:
IMPRESSION:   
1. No findings suggestive of osteomyelitis. 
2. L4-L5: Multifactorial narrowing of the subarticular zones and 
neural 
foramina, with impingement upon the traversing L5 and the 
exiting L4 nerve roots, respectively. 

"""

gold_text = """
Mr. ___ was evaluated in the Emergency Room and admitted to 
the SICU for further management. In the ED he had a his chest 
xray had new LUL opacities and RLL opacities concerning for 
pneumonia. He had an NGT placed with rapid outflow on about 1 
liter of bilious fluid. His WBC was elevated with a left shift. 
BUN and CR were 45/2.2 respectively. He was seen by transplant 
for his epigastric pain who recommended a CT of the torso which 
showed new central left upper lobe and right lower lobe airspace 
consolidation with near complete resolution of the left lower 
lobe airspace consolidation and probable loculated right pleural 
effusion as well as improved but mildly persistent chronic 
phlegmon with few residual foci of intrinsic air noted extending 
from the inferior stomach antrum towards the anterior abdominal 
wall with decreased amount of intrinsic, but extraperitoneal air 
in this phlegmonous change. He was seen by renal who did not 
recommend dialysis at the time. He was continued on tacrolimus 
2.5 with daily tacrolimus levels and started empirically on 
linezolid and cefepime for a history of VRE. He was started on a 
heparin gtt and his coumadin was held. His INR was 
supratherapeutic so it was reversed with FFP and Vitamin K. 
Transplant surgery recommended EGD to rule out a perforation in 
the area of his previous perforated duodenal ulcer. Sputum gram 
stain found GPR and GPC's. On HD 2 he continued to make good 
urine, his sats were in the high 90's on 2L. He was started on 
flagyl for broader spectrum coverage. His linezolid was 
discontinued and he was started on vancomycin. His heparin gtt 
was held in anticipation for bronchoscopy however he did not get 
bronched until HD3 which revealed significant mucopurulent 
secretions throughout the airway with mucous plugs and moderate 
supraglottic edema. BAL cultures were sent which showed no 
growth. His heparin gtt was restarted. EGD to evaluate his 
epigastric pain was planned but was difficult to coordinate due 
to his need for anesthesia and potential respiratory 
instability. He was seen and evaluated by physical therapy and 
then transferred to the floor. On HD 4 he had an episode of 
desaturation to the upper 80's which improved with suctioning. 
His Cr had remained stable at 2 for several days so HD continued 
to be deferred. His WBC count decreased to 9.6. On HD 5 he was 
no longer reporting abdominal pain so his NGT was discontinued. 
On HD 6 his creatinine increased to 2.3 from 2 and his diet was 
advanced to sips. Because his Cr continued to remain elevated on 
HD 7 he was started on a bicarb drip. His diet was advanced to 
clears per transplant recs. His trach was capped after which he 
experienced an episode of desaturation on room air which 
improved with uncapping and trach mask. His tacrolimus dose was 
decreased to 2mg BID per renal recs for a tacro level of 11 up 
from 5. On HD 8 (___) his heparin gtt was held for a 
bronchoscopy and EGD which were performed in the operating room. 
He tolerated the procedures well and was returned to the floor 
in stable condition. Bronchoscopy found thick whitish yellow 
secretion in the trachea and bilateral lungs, more on the right 
side. Therapeutic aspiration was performed and a washing found 
commensal respiratory flora. EGD found moderate erythema and 
scattered ulceration in mid esophagus, scattered erosions in 
body of the stomach consistent with NG tube trauma, and a small 
mucosal defect in D1 at site of the prior surgical anastomosis. 
The remainder of duodenal mucosa was normal. After the procedure 
he was restarted on coumadin with a heparin gtt bridge for 
treatment of a prior IJ thrombus and his diet was advanced to 
regular. On HD 9 his creatinine rose to 2.5. His urine output 
continued to be adequate and renal did not wish to start 
dialysis. He was also noted to have what was thought to be 
increased psychomotor retardation likely secondary to 
depression. Psychiatry was consulted and had no new 
recommendations. He continued to have copious secretions 
requiring suctioning throughout the day. His trach mask was 
weaned from 60% to 40% FiO2 with capping attempted throughout 
which he only tolerated for brief periods before becoming 
tachypneic. On HD 10 he was triggered for unresponsiveness and 
inability to cooperated with an exam. He was moving his upper 
extremities. ABG revealed a metabolic acidosis with hypoxia and 
he was transferred to the SICU where his blood gas and 
respiratory status improved on BiPAP. An NGT was placed to 
provide him with medications and nutrition. Because his INR was 
therapeutic his heparin gtt was stopped. Neurology was consulted 
thought that his symptoms were more fitting with a metabolic 
disturbance rather than cerebrovascular in origin. His 
tacrolimus level was decreased to 1.5 BID. Though his 
respiratory status was somewhat improved he continued to have 
altered mentation and an elevated Cr so on HD ___ he underwent 
hemodialysis. Because BAL from ___ was negative for pathogenic 
flora his cefepime and flagyl were discontinued. His diet was 
advanced to regular though his intake was poor. His mental 
status continued to improve and he was much more alert and 
interactive. Chest xray showed slightly enlarging bilateral 
plural effusions so on HD 12 he underwent bronchoscopy which 
noted significant supraglottic edema. The tracheal button was 
noted to be partially occluding the trachea depending on his 
head position so it was repositioned. Minimal granulation tissue 
seen with a patent airway. Significant mucus plugs were seen in 
the distal airways which were removed. BAL was noted to have 
2+GPC on gram stain. He was complaining of difficulty swallowing 
which was thought to be related to his supraglottic edema so he 
was restarted on tube feeds through his NGT. On HD ___ he was 
started on Cefepime for a WBC that had increased to 13.2 and the 
positive gram stain from the day prior. On HD 14 his WBC count 
continued to rise to 14.7 so an ID consult was obtained. They 
recommended multiple cultures including c-diff, CMV and 
legionella all of which came back negative. On HD 15 his 
cefepime was stopped with a plan to reculture him if he spiked a 
fever or if his WBC increased. He was seen by ENT for his 
supraglottic edema and laryngoscopy at the bedside showed mild 
epiglottic and arytenoid edema. Over the next several days his 
WBC normalized and his secretions improved, requiring less 
frequent suctioning. On HD 18 he had an episode of desaturation 
to the 60's overnight which improved with trach mask. He was 
noted to have bilateral lung opacities on CXR. Because of his 
episode of desaturation and his known prior positional occlusion 
of his tracheal button, his warfarin was held and his button was 
exchanged with IP for a tracheostomy tube. After the exchange he 
was maintained on pressure support with minimal settings. The 
following day his warfarin was restarted with a heparin gtt, his 
tube feeds were restarted through a Dobbhoff which was exchanged 
for his NGT. He was also ordered for chest ___ with a therapeutic 
shaking vest to help clear his secretions however he refused its 
use. On HD 20 he was given a unit of PRBC for a Hct of 21.6 with 
improvement to 25.7. Over the next several days TF advanced to 
goal at an increased caloric density/reduced volume after 
experiencing some nausea and multiple attempts were made to wean 
him off of the vent, however each time he was transitioned to 
the trach mask he became tachypneic requiring that he be put 
back on pressure support. It was thought that these episodes 
were due to anxiety so psychiatry was asked to weigh in. On HD 
22 his celexa was increased and he was restarted on low dose 
ativan per psychiatry recommendations. Because he continued to 
have high tube feed residuals on HD ___ his dobhoff was advanced 
past the pylorus in the ___ suite. His tunneled HD line was 
removed by ___ as well and he was given 10 IV lasix for a CXR 
that looked wet. Over the next several days his Cr remained 
stable however his K rose as high as 5.9 on HD 25. 30 of IV 
lasix was given and his repeat K was down to 5.3. His tube feeds 
were changed from Isosure to Nephro in order to reduce the 
amount of K in his diet. His dobhoff was also noted to not be 
working and was unable to be unclogged so it was exchanged. Per 
renal recommendations he was started on 20 of IV lasix. He had 
been tolerating trach mask for over 24 hours and was tolerating 
a PMV so he was transferred back from the ICU to the surgical 
floor. His K increased to 5.5 and he was given an additional 20 
of IV lasix after which his K was reduced to 5. He was stable on 
the floor on a trach mask and on HD 27 he was noted to have a 
Hct of 21.5 which was thought to be secondary to anemia of 
chronic disease. He was given 2 units of PRBC's with a post 
transfusion Hct of 27.8. His Hct remained stable over the next 
several days. On HD ___ he was set up with an inexsufflator which 
he was to use daily, with moderate success. The following day he 
self d/c'd his NGT. He passed a swallow evaluation however he 
continued to refuse to take much by way of PO intake. On HD 30 
his dobhoff was placed past the pylorus under flouroscopic 
guidance and his tube feeds were restarted. He was also started 
on epoetin 4000U 3x/week. The following day his standing lasix 
were stopped as he was thought to be volume down with a BUN in 
the upper 60's. On HD 32 his tube feeds were cycled over 18 
hours rather than given continuously because he was complaining 
of vague abdominal pain. His K was 5.6 so per renal 
recommendations he was given 1.5 amp of HCO3 in 500 of D5W along 
with 20 of IV lasix with subsequent improvement in his K to 5.2. 
The same process was repeated again the following day because of 
a K of 5.6 with improvement to 5.2. Over the next several days 
his tube feeds were switched from cycling to continuous and the 
back to an 18 hour cycle in an effort to help with his vague 
abdominal pain. He was started on megase in an effort to help 
with his appetite as he continued to take in minimal PO. 
He continued to have problems with hyperkalemia, requiring 
occasional kayexelate.  His diet order limited potassium to 1 
gram but he was also eating food brought in by his family and 
this could not be restricted.  The renal service followed him 
closely and did a battery of blood work and urine assays.  Still 
pending is renin and aldosterone levels.
His blood sugars were also elevated to the 300 range as he took 
both tube feedings and and a diabetic diet.  His Lantus was 
increased to BID and the sliding scale was tightened up.

(*********CURRENT THROUGH OF ___

The patient was transferred to the medicine service for 
management of his hyperkalemia and hyperglycemia and a summary 
of his medical issues after his transfer are detailed below.  I 
will try to incorporate some of the complexity of his course 
prior to our transfer.

# Hyperkalemia: At the time of transfer his potassium was 
between 5.3-6.8.  The source of his hyperkalemia was unclear.  
He was being titrated up on his insulin, he was being diuresed 
with lasix, which should lower his potassium.  However, he had 
been started on tube feeds and also was persistently acidotic, 
which could raise his potassium.  A renin level was sent that 
was 1.51 (within the normal range) and aldosterone level was 
high at 31.  The high aldosterone level, one would expect a low 
potassium.  He was having normal bowel movements.  It was 
largely thought that most his elevated potassium levels were 
related to both his dietary intake and also his heparin 
injections.  We talked to nutrition about his tube feeds and 
lowered there potassium content.  In addition, we limited his PO 
potassium intake and stopped his heparin injections as heparin 
can cause a hyperkalemia.  His elevated tacro can also cause a 
mild hyperkalemia.  We also attempted to fix his acidemia with 
NaHCO3 tablets.  After making some of these adjustments (he 
remained persistently acidemic) his potassium level normalized 
and was slightly elevated to 5.4.  

# IDDM/hyperglycemia: When the patient was transferred to our 
service his FSG were persistently elevated.  Going through his 
medications, there were not medications that elevated his 
potassium and we made sure that all IV medications were mixed in 
normal saline and not D5W as they often are.  In addition, he 
was on a regular diet and so it was changed to a diabetic 
consistent carbohydrate diet.  We also spoke with nutrition 
about adjusting his tube feeds.  He was eating doughnuts and 
other food from his wife and we spoke with his about keeping up 
a consistent carbohydrate diet.  His ISS and long acting 
Glargine were adjusted and his sugars became better controlled 
for some time.  Later in his stay, he had elevated sugars of 
unclear etiology.  During that period it was discovered he had a 
pseudomonal UTI.  We began treatment, but he continued to have 
elevated am FSG in the setting of adjusting his tube feeds.  We 
calculated his am humalog requirements and increased his 
nighttime glargine by ___ the total am requirement.  We also 
started monitoring a 3am FSG in the middle of his tube feeds so 
that we could have better control while he is being fed and 
receiving the majority of his daily calorie intake.  After 
titrating his medication his sugars were better controlled and 
he was discharged on a sliding scale with Glargine 34 in the am 
and 60 in the pm.  He will need to have his fingersticks 
monitored closely at the LTACH and his insulin adjusted as he 
weans himself off tube feeds and increases his oral intake.  If 
continues to be elevated you could consider endocrine consult.  

# Leukocytosis/thrombocytosis:  When the patient was transferred 
to our service he had a reactive thrombocytosis, but his white 
count was normal.  In the setting of uncontrolled hyperglycemia 
and a reactive thrombocytosis for 4 days, we were concerned for 
an infection.  Two days after transfer, he developed a white 
count and a U/A was sent off, stool cultures were sent and 
patient had a CXR.  His infectious work up at the time was 
negative.  He had a fluctuating white count for 7 days 
(___).  During that time sputum was also collected and 
the patient was bronched that did not grow any bacteria.  There 
was discussion about imaging his coccyx where there was  an 
ulcer and potential concern for osteo.  His RLE wound was also a 
potential source.  ID was consulted and the plan was to hold off 
on further imaging and abx until the patient declared himself.  
He had other reasons for a leukocytosis and it was not clear the 
leukocytosis was related to an infection.  B-glucan was also 
sent that was the upper limit of normal and galactomannan was 
negative.  Blood cultures were sent and negative and histoplasma 
and CMV were also sent and returned negative.  On ___ his 
WBC normalized then increased on the ___.  Repeat work up 
revealed a UTI and ceftriaxone was started.  On the ___ his 
white count almost doubled to 25.  At that time patient had a CT 
chest, CT and/pelvis and MRI of the coccyx that did not reveal 
another source of infection.  In addition, rpt C. dif was sent 
and blood cultures drawn that were also negative.  The patient 
was broadened to vanc/ceftriaxone and flagyl temporarily, and 
then the ceftriaxone was changed to cefepime when his urine came 
back with pseudomonas.  When the final reads of the images came 
back and did not reveal an intrabdominal process or osteo of the 
sacrum, the vancomycin and flagyl was discontinued.  Because of 
the UTI, his foley was removed, but the patient was retaining 
and so it was ultimately replaced.  On ___, I spoke to the 
patient and he agreed to straight cath Q8H.  In addition, once 
he was started on cefepime, his WBC started trending down.  He 
will need a 14 day course of cefepime for his UTI to be 
completed on ___.

# UTI: As above, he was initially started on ceftriaxone and 
once the cultures came back, he was changed to cefepime.  Plan 
is for a 14 day course of cefepime.  Last Day of cefepime will 
be ___.  This was a catheter associated UTI.  The patient 
is thought to have a neurogenic bladder and the plan initially 
was for long term foley placement.  However, given his risk for 
recurrent UTIs, I spoke with the patient and nursing about 
straight catheterization Q8H or sooner if his bladder scan is 
greater than 500cc on a regular basis.  The patient was 
agreeable to this plan.  

# S/p Gastric perforation in ___ and repair:  He has a 
persistent phlegmon anterior to the site of his repaired ulcer.  
It has a persistent sinus tract and could be a nidus for 
infection, but repeated imaging on multiple occasions reveals 
stable to slightly improving phlegmon.  It is being monitored 
closely and should be considered as a potential source of 
infection.

# Pressure ulcers: Mr. ___ has multiple pressure ulcers.  
Despite frequent turning, he developed a right scapular pressure 
ulcer that has healed well.  He also has a decubitus ulcer and a 
RLE lateral shin ulcer (he sits in a frog leg position).   With 
regards to his RLE ulcer, it was healing well with pink 
granulation tissue, but the patient had pneumoboots on at the 
time of transfer.  The pressure of the pneumoboots caused a 
hematoma to develop around the ulcer.  we had wound care come 
see the wound and make recommendations.  We continued aggressive 
wound care and he had multipodius boots on bilaterally.  Despite 
daily dressing changes and offloading the weight of the RLE, he 
developed a surface eschar over the wound.  Surgery came to 
debride the wound and it was stage II/III with pink granulation 
tissue under the eschar.  Wound care recommended offloading the 
weight of the right leg with soft pillows.  The left foot should 
remain in multipodus boots.  As for his coccyx ulcer it has 
progressed since he was transferred to medicine.  Initially, he 
had to separate stage II ulcers.  One at the superior portion of 
the anal fissure and the other above the coccyx.  It was 
appeared to be getting smaller initially.  The patient began 
having multiple bowel movements and it was hard to keep the 
stool out of the wound.  The area was cleaned regularly, but 
given the location, it was at times covered in stool.  When the 
patient's white count started rising there was concern that 
there could be osteo of the sacrum.  MRI was ordered and was 
negative for osteo.  The 2 ulcers coalesced and he now has one 
large ulcer that developed a boggy eschar.  Surgery was 
consulted and felt the area did not need to be debrided.  The 
area will have to be cleaned regularly and he will need to be 
turned Q2H to ensure offloading of his weight.  He should be up 
and out of bed 3 times a day.

# ESRD s/p renal transplant: Mr. ___ was admitted with a 
creatinine of 2.2.  Over the course of his stay his creatinine 
fluctuated down to 1.1 and then up to 1.6.  Initially his ___ 
was likely from dehydration.  When his creatinine began to rise 
on ___ his white blood cell count was also rising and work 
up revealed an UTI and he had also been overdiuresed.  It was 
felt his ___ was mostly related to dehydration.  He was given 
fluids and blood (as described below) for intravascular volume 
and his creatinine resolved.  Renal was following for most of 
his hospital stay.  He was continued on his tacrolimus, MMF and 
prednisone.  His MMF and prednisone doses were stable, but his 
tacrolimus doses were repeatedly adjusted depending on his 
levels.  Given his anemia and concern for anemia of chronic 
inflammation and also low epo levels, he was started on EPO 
4,000 3x/week and also transfused as needed.  when his euvolemia 
was maintained, his renal function improved and on discharge he 
was making good urine and his creatinine was 1.2. 

# Pulmonary secretions/dyspnea:  The patient was initially 
admitted on ___ for increased secretions, respiratory 
distress and hypoxia.  He was felt to have a multifocal 
pneumonia and was given a course of vancomycin, cefepime and 
levofloxacin.  He had multiple bronchoscopies that did not grow 
any bacteria.  On bronchoscopy and repeated imaging he does have 
chronic mucous plugging in the RLL.  IP had removed the mucous 
plugs on a few occasions, but it seems to recur.  He had 
persistent secretions and was given guaifenesin w/ codeine on a 
PRN basis as a mucolytic.  He also had a fan in front of him as 
cool air helped relieve his dyspnea. We used humidified air with 
the trach mask to also help with airway passage into the lungs.  
Respiratory therapy saw him regularly and he does have 
significant secretions, but there is also an anxiety component 
to his dyspnea.  He was started on clonazepam 0.25mg PO BID for 
anxiety and he is also written for ativan on a PRN basis.  The 
patient was encouraged to get up and out of bed and participate 
in chest ___ as well as use the insuffulator.  He was a little 
resistant, but with encouragement would allow some chest ___.  He 
was out of bed at least 1 time per day.  His lung sounds have 
continued to improve and on the day of discharge.  

# Anemia:  Mr. ___ had persistent anemia throughout his 
admission, requiring multiple blood transfusions.  His guaiac 
have been negative and he has known renal disease and was 
started on EPO 4,000 units 3x/week.  He also has anemia of 
chronic inflammation.  His Hemoglobin hovers in the ___ range, 
but he will need transfusions on an as needed basis.  At the 
time of discharge his last transfusion was on ___.  He will 
need weekly CBC and transfuse for a hemoglobin less than 7.  
There were no signs of hemolysis onlaboratory values.  

# Hypercalcemia:  Patient was noted to be hypercalcemic on 
___.  PTH was ___, which did not significantly explain her 
hypercalcemia.  His vitamin D level was low at 17. He was 
started on Vitamin D 50,000 units Q weekly and will need a 
repeat level in 8 weeks.  His vitamin D regiment should be 
adjusted as per that level, but will likely need 1,000 units per 
day.  His calcium normalized, but when corrected with his low 
albumin (3.0), it was still mildly elevated.  

# H/o right IJ thrombosis and CVA: Patient is being 
anticoagulated with warfarin.  Given his risks for clots, he was 
bridged when he was subtherapeutic.  On 6mg, he had a rapid rise 
in his INR and so his INR was lowered to 5mg the day prior to 
discharge.  His INR will need to be monitored closely and 
coumadin doses adjusted so that he can have a stable INR with 
the goal between ___.  He will be discharged on 5mg and will 
need a repeat INR on ___ and need to adjust his dose based 
on his level

# Anxiety/depressions: Given his prolonged hospital stays and 
clinical deterioration over the last 5 months, the patient is 
notable very depressed and anxious.  He was seen by psychiatry 
who felt this could be adjustment disorder vs. a true 
depression.  Social work is actively following and speaking with 
the family and the patient.  He was started on citalopram during 
this admission and his dose was steadily increased to a maximum 
dose of 40mg PO Daily.  In addition, standing clonazepam was 
started and he received ativan PRN and gets it ___ times per 
day.  Hopefully when he is discharged his strengths steadily 
improves at the LTACH and his mood improves when his symptoms 
resolves.   He should be followed by social work and psychiatry 
in the outpatient setting.

# Acidosis:  stable in the ___ range.  It improved when he 
takes his bicarb, however, he intermittently refuses and has not 
taken ir over the last 36 hours.  I spoke with him at length 
about the importance of this medication and he agreed to take 
it.  It is important to ensure that he continues to take his 
sodium bicarbonate.

# FEN/GI: Pt has been struggling with taking in food by mouth.  
He had to have tube feeds started during this hospitalization 
and while he intermittently take in some food orally, nutrition 
has been calorie counting and the amount is minimal.  We tried 
decreasing his tube feeds in order to create a situation where 
he would be hungry and want to eat, but this resulted in poor 
calorie intake and his tube feeds were started again.  His 
eating was complicated by intermittent nausea of unknown 
etiology.  His catscans were negative for obstruction.  He is 
having regular BM.  The Dobhoff is past the pyloric sphincter.  
He is tolerating his tube feeds currently.  A long discussion 
was had with the patient about the risks of leaving in a Dobhoff 
and the need for a G-tube.  He adamantly refuses a G-tube.  His 
wife also does not want a G-tube and feels that once he starts 
working with ___, he will have a bigger appetite and will eat 
more by mouth and we will be able to wean him off his tube feeds 
and back on a regular diet.  

TRANSITIONAL ISSUES:
LABWORK:
___
tacro levels to Dr. ___ at ___: ___
- Will need CBC 3x/ week and transfuse for Hct less than 21
- Daily BMP, Ca, Mg, P for now and trend potassium as tacro 
levels come down, potassium should come down as well
- Trend White blood cell count (3x/week) as it has stablized, 
but if trending up would need to have further work up.
- Next INR check on ___ and adjust coumadin as needed

VOLUME STATUS:
- Monitor his fluid status closely.  If concerned for 
dehydration or BUN/Cr rising on labs, would give 1L NS
- Make sure he is taking his Sodium Bicarbonate

ANTIBIOTICS:
- Cefepime 1gm Q12H until ___

GENERAL CARE:
- Straight Cath Q6H
- Wound care daily
- frequent turning Q2H
- Will need aggressive ___ and OT
- Chest Physical therapy
- Continued suctioning and trach care
- Tube Feeds unless patient is able to be transitioned to PO or 
agrees to G-tube
- need to follow up on his beta glucan
- If FSG are persistently elevated would have endocrine come to 
see him

DEPRESSION/ANXIETY:
- Would increase clonazepam to 0.5mg BID if patient is requiring 
a lot of ativan
- Consider psych consult if patient is not participating in care
- Social worker and case managers to follow closely
- Needs lots of encouragement.
"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"

prompt = "You are a medical assistant. Your task is to write the brief hospital course corresponding to the following hospital discharge:\n\n{}\n\nBrief Hospital Course:".format(input_text)

model_path = "/pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2-extended"

kwargs = {
    "model": model_path,
    "tokenizer": model_path,
    "trust_remote_code": True,
    "max_num_seqs": 1,
    "tensor_parallel_size": torch.cuda.device_count(),
    "dtype": 'bfloat16',
    "gpu_memory_utilization": 0.5
}

client = vllm.LLM(**kwargs)

print(f"vLLM client initialized")
BOS_TOKEN, EOS_TOKEN = '<|im_start|>', '<|im_end|>'

GREEDY_PARAMETERS = {
    'best_of': 1,
    'presence_penalty': 0.0,
    'frequency_penalty': 1.0,
    'top_k': -1,
    'top_p': 1.0,
    'temperature': 0.0,
    'stop': EOS_TOKEN,
    'use_beam_search': False,
    'max_tokens': 6144,
}

sampling_params = vllm.SamplingParams(**GREEDY_PARAMETERS)
response = client.generate(prompt, sampling_params=sampling_params)

print(response)

# print len of tokenized input
print(f"Tokenized input length: {len(response[0].prompt_token_ids)}")

# print len of tokenized output
print(f"Tokenized output length: {len(response[0].outputs[0].token_ids)}")

cumulative_logprob = response[0].outputs[0].cumulative_logprob
perplexity = np.exp(-cumulative_logprob / len(response[0].outputs[0].token_ids))

print(f"Perplexity: {perplexity}")

if len(response) > 0:
    answer = [r.outputs[0].text for r in response]
else:
    answer = response[0].outputs[0].text

print(answer)

