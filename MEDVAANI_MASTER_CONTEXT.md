# MEDVAANI — MASTER PROJECT CONTEXT & BUILD PLAN
## Canonical context document for AI-assisted development, design, testing and competition work

**Project:** MedVaani  
**Type:** Voice-first pharmaceutical Medical Information & Adverse Event reporting platform  
**Primary demo product:** Lantus® SoloStar® (insulin glargine, rDNA origin) Injection  
**Primary demo manufacturer/reference context:** Sanofi  
**Purpose:** Great Indian AI Internship competition + AI Product Management portfolio project  
**Document role:** This is the canonical source of truth for the project. AI coding/design assistants should use this document as context before making project decisions.

---

# 1. NON-NEGOTIABLE PROJECT DEFINITION

MedVaani is a **voice-first pharmaceutical Medical Information (MI) and Adverse Event (AE) reporting platform**.

Its purpose is to create a safe conversational access layer between a pharmaceutical product and a person who:
1. needs approved product information, or
2. wants to report a possible adverse event/side effect.

MedVaani is NOT:
- an AI doctor
- a symptom checker
- a diagnosis tool
- a treatment recommendation engine
- a medication/dose adjustment tool
- a hospital finder
- a generic ChatGPT-style medical assistant
- an autonomous pharmacovigilance decision maker

The system must remain bounded to its intended pharmaceutical-information and safety-reporting workflows.

---

# 2. CORE PRODUCT PROMISE

Primary positioning:

> **From a patient's voice to a structured pharmacovigilance case.**

One-line product description:

> **A voice-first pharmaceutical Medical Information & Adverse Event reporting platform.**

Tagline:

> **Ask. Report. Stay Informed.**

The product should feel like a modern healthcare product with conversational AI, not like a generic chatbot.

---

# 3. PRIMARY USER EXPERIENCE

The user reaches a medication-specific MedVaani page.

Current prototype medication:

> **Lantus® SoloStar®**  
> Insulin glargine (rDNA origin) Injection

Landing-page actions:

### A. Get Product Information
Conversational MI experience grounded in approved/public product information.

### B. Report a Side Effect
Conversational AE intake experience that converts a user's description into a structured safety report.

Both workflows support:
- typed text
- voice input
- multilingual/Indian-language conversation
- code-switching where supported
- safety guardrails

The AE workflow is the hero feature for the competition demo.

---

# 4. PRODUCT ENTRY EXPERIENCE

Conceptual flow:

Product / QR / short URL
        ↓
MedVaani Lantus page
        ↓
User chooses:
  • Get Product Information
  • Report a Side Effect
        ↓
Conversational interface
        ↓
Text OR Voice
        ↓
Safety guardrail
        ↓
MI or AE workflow
        ↓
Structured response / case
        ↓
Human review where appropriate

IMPORTANT:
The prototype must NOT claim that MedVaani is officially integrated with Sanofi or that a real QR code, pharma database, or safety system is connected unless such integration actually exists.

Use public/reference information and clearly label the prototype/demo nature.

---

# 5. BRAND & VISUAL LANGUAGE

Brand:
> **MedVaani**

Logo direction:
- "Med" bold deep navy
- "Vaani" lighter charcoal/black
- small 3-stroke voice/spark mark
- orange/pink/purple gradient accent
- white background

Visual reference:
Modern, minimal, premium healthcare-tech aesthetic inspired by the soft pastel visual language seen in the Gnani website.

Design principles:
- white/cream base
- pastel peach/pink/purple/blue gradients
- soft glow
- generous whitespace
- rounded cards
- subtle borders
- clean typography
- restrained shadows
- accessible contrast
- professional healthcare feel
- no excessive futuristic decoration

Avoid:
- cartoon healthcare graphics
- giant medical crosses
- cluttered dashboards
- dark cyberpunk styling
- excessive neon
- generic AI robot imagery

---

# 6. PRIMARY INTERFACE STRUCTURE

The main conversational application should contain:

### Header
- MedVaani logo
- Lantus® SoloStar® product identity
- language selector
- user/profile control

### Left navigation
- New Chat
- Get Product Information
- Report a Side Effect
- Previously Reported Adverse Events

Do NOT label this section "Chat History" in the AE workflow.

### Main content
Conversational reporting interface.

### Safety panel
A visible but compact emergency-help panel:
> If you are experiencing a serious or life-threatening reaction, seek immediate medical attention.

### Safety disclaimer
Example:
> This is not medical advice. MedVaani provides product information and helps document safety reports. Consult a qualified healthcare professional for personal medical guidance.

---

# 7. AE REPORTING — CANONICAL UX

The AE experience must feel like a guided conversation rather than a giant form.

Example beginning:

MedVaani:
> I'm sorry to hear you're experiencing this. I can help you report it. To get started, could you describe what happened?

User:
> I am having a headache after taking Lantus.

MedVaani:
> Thank you for sharing that. I'll collect some basic information to create your safety report.

Then progressively collect required information.

The AI should NOT dump 15 questions at once.

---

# 8. AE DATA MODEL

## Core valid-case concept

The prototype should understand that a pharmacovigilance case fundamentally needs:

1. identifiable reporter
2. identifiable patient
3. adverse event/reaction
4. suspect product

These are the core ICSR concepts.

Do NOT claim that every jurisdiction universally requires every field below.

---

# 9. MEDVAANI AE INTAKE POLICY

For the prototype, the following are treated as required or essential for the intended intake flow.

### Patient/reporting identity
- patient first name
- patient last name
- date of birth OR appropriate age/identifier
- sex/gender field according to the prototype's chosen data policy
- reporter relationship to patient
- reporter first/last name when reporter is not the patient
- contact number
- email
- preferred contact method
- consent to contact later

### Event
- suspect product
- adverse event description
- onset date/time or approximate timing
- current status/outcome

### Product
- product name
- formulation/strength where available
- dose, when available

Important:
These are the prototype's intake requirements, NOT a claim that every field is universally mandated by pharmacovigilance regulation.

---

# 10. OPTIONAL AE INFORMATION

Ask after essential information has been collected and allow the user to skip.

Potential optional fields:
- therapy start date
- indication
- dose/frequency
- concomitant medications
- medical history
- allergies
- physician/prescriber details
- weight
- height
- relevant laboratory values
- treatment/action taken
- hospitalization/medical attention
- outcome details
- lot/batch number
- expiry date

The UI should explicitly communicate that additional questions can be skipped where appropriate.

---

# 11. PRIVACY PRINCIPLE

Do not over-collect personal data merely because it is possible.

The product should prefer:
- minimum necessary data
- clear purpose
- transparent contact consent
- secure handling
- human review where needed

Avoid forcing a full home address into the initial AE conversation unless a specific regulatory/company workflow requires it.

If address is shown in the prototype, make clear that this is a configurable workflow field, not a universal regulatory requirement.

---

# 12. REPORTER LOGIC

At an appropriate point, ask:

> Are you reporting this for yourself?

If YES:
- patient = reporter
- avoid asking duplicate identity information

If NO:
- collect reporter identity/contact information separately
- retain patient information separately

The conversation should not make the user repeat the same information unnecessarily.

---

# 13. AE CONVERSATION STATE MACHINE

The backend should own the workflow state.

Recommended states:

CASE_CREATED
↓
PRODUCT_IDENTIFIED
↓
EVENT_IDENTIFIED
↓
PATIENT_IDENTIFIED
↓
REPORTER_IDENTIFIED
↓
CONTACT_CAPTURED
↓
CONSENT_CAPTURED
↓
EVENT_TIMING_CAPTURED
↓
SERIOUSNESS_SCREENED
↓
OUTCOME_CAPTURED
↓
OPTIONAL_DETAILS
↓
CASE_REVIEW
↓
CASE_COMPLETED / HUMAN_REVIEW

The LLM should NOT independently control the entire workflow.

Python/FastAPI should control:
- required fields
- state transitions
- validation
- safety interruption
- completion
- case status

Evon should assist with:
- intent detection
- information extraction
- interpretation of natural-language responses
- question generation within the current workflow state
- summarization

---

# 14. SAFETY GUARDRAIL — ALWAYS ACTIVE

Safety is not a separate optional screen.

The safety layer must operate in BOTH:
- Medical Information workflow
- Adverse Event workflow

Architecture:

USER
 ↓
Voice → Prisma
Text → direct
 ↓
Evon
 ↓
ALWAYS-ACTIVE SAFETY GUARDRAIL
 ↓
MI OR AE WORKFLOW
 ↓
RAG / STRUCTURED CASE
 ↓
RESPONSE
 ↓
Voice → Timbre

---

# 15. MI SAFETY RULES

MedVaani may:
- answer approved product-information questions
- explain information contained in the supplied product reference
- provide general product facts grounded in the reference

MedVaani must NOT:
- diagnose
- prescribe
- recommend personalized treatment
- recommend changing a medication dose
- recommend stopping/starting medication
- make clinical decisions
- interpret a patient's condition as a diagnosis

Example:

User:
> My blood sugar is 250. Should I increase my Lantus dose?

Safe behavior:
> I can provide information about Lantus based on the product information available to me, but I can't recommend a personalized insulin dose or treatment change. Please speak with your doctor or healthcare professional for advice specific to your situation.

---

# 16. AE SAFETY RULES

AE reporting is not medical diagnosis.

The system should:
- document what the user reports
- ask relevant safety-reporting questions
- detect potentially urgent language
- interrupt the questionnaire when urgent symptoms are detected
- recommend seeking urgent medical attention when appropriate
- flag the case for human review
- preserve the report information already collected

It must NOT:
- diagnose
- minimize symptoms
- claim the drug caused the event
- tell the user to change medication
- provide unsupported treatment instructions
- automatically decide causality
- autonomously close a serious case

---

# 17. URGENT SAFETY INTERRUPT

Potential trigger examples:
- severe difficulty breathing
- unconsciousness
- severe chest pain
- severe bleeding
- other clearly urgent/life-threatening descriptions

Behavior:

1. stop normal questioning temporarily
2. clearly state that urgent medical attention may be needed
3. advise contacting local emergency medical services / seeking immediate care
4. flag the case as high priority
5. allow safety-reporting workflow to continue only when appropriate

Example:

> What you're describing may require urgent medical attention. Please seek immediate medical care or contact your local emergency service. I can also help document the event for a safety report once it is safe to continue.

Do not diagnose anaphylaxis or another condition unless the task is specifically supported by authoritative reference and the wording remains appropriately bounded.

---

# 18. MI RAG ARCHITECTURE

Reference:
> Publicly available Lantus/Lantus SoloStar product information / monograph.

Pipeline:

PDF
 ↓
PyMuPDF extraction
 ↓
clean text
 ↓
chunking
 ↓
embeddings
 ↓
FAISS vector store
 ↓
query
 ↓
top-k retrieval
 ↓
Evon
 ↓
answer constrained to retrieved reference

Evon instruction:
> Answer only from the supplied reference context. If the answer is not supported by the retrieved material, say that sufficient information is not available rather than inventing an answer.

The system must avoid hallucinated medical information.

---

# 19. VOICE ARCHITECTURE

Voice path:

User speaks
 ↓
Gnani Prisma v2.5
 ↓
transcription
 ↓
FastAPI
 ↓
Evon v3.3
 ↓
Safety + workflow/RAG
 ↓
response text
 ↓
Gnani Timbre v2.5
 ↓
audio response

Voice should support:
- English
- Hindi
- Indian languages where supported
- Hinglish/code-switching
- natural conversational speech
- noisy/telephonic-style testing where feasible

Voice should not simply be a voice version of a form.

---

# 20. CHAT INTERFACE

The chat AE screen should visually communicate:

- reporting a side effect
- safety disclaimer
- assistant messages
- user messages
- extracted information
- current progress
- input box
- microphone button
- send button
- emergency-help panel

Example:

Assistant:
> I'm sorry to hear you're experiencing this. I can help you report it. What happened?

User:
> I am having a headache after taking Lantus.

Assistant:
> Thank you. I'll collect some basic information for your safety report. What is your first and last name?

User:
> [name]

Assistant:
> Thank you. What is your date of birth?

...

The chat should feel like a calm guided interview.

---

# 21. VOICE INTERFACE

Voice UI should retain the same overall layout but make speech the primary interaction.

Key elements:
- large microphone control
- animated waveform
- listening state
- transcription preview
- confirmation buttons
- retry recording
- switch to keyboard
- language selector

Example:

User speaks:
> I am having a headache after taking Lantus.

System:
> I heard: “I am having a headache after taking Lantus.”  
> Is that correct?

Buttons:
- Yes, that's correct
- No, let me try again

This confirmation is particularly useful for medical terminology, names, numbers and dates.

---

# 22. PREVIOUSLY REPORTED ADVERSE EVENTS

Navigation item:

> **Previously Reported Adverse Events**

NOT:
> Chat History

This area is for safety-report records, not generic conversation history.

Example case cards:

AE-2026-00124
- Product: Lantus SoloStar
- Event: Headache
- Reported: 02 Oct 2026
- Status: Submitted

AE-2026-00118
- Product: Lantus SoloStar
- Event: Injection-site redness
- Reported: 21 Sep 2026
- Status: Under Review

Use fictional demo data only.

---

# 23. REVIEWER DASHBOARD

The prototype may include a separate reviewer view.

Purpose:
Show how an AI-generated conversational report becomes a structured pharmacovigilance case.

Dashboard fields:
- Case ID
- Product
- Event
- Reporter type
- Onset
- Seriousness
- Outcome
- Missing information
- Priority
- Status
- Created timestamp

Potential statuses:
- Draft
- Ready for Review
- Human Review
- Submitted
- Follow-up Required

The AI should assist the reviewer, not replace pharmacovigilance professionals.

---

# 24. STRUCTURED CASE OUTPUT

Example internal representation:

{
  "case_id": "AE-2026-00124",
  "product": "Lantus SoloStar",
  "intent": "adverse_event_report",
  "patient": {
    "identifier": "...",
    "dob_or_age": "...",
    "sex_or_gender": "..."
  },
  "reporter": {
    "relationship": "patient",
    "name": "...",
    "phone": "...",
    "email": "..."
  },
  "event": {
    "description": ["headache"],
    "onset": "...",
    "seriousness": "unknown",
    "outcome": "unknown"
  },
  "consent_to_contact": true,
  "missing_information": [],
  "priority": "standard",
  "status": "ready_for_review"
}

Never use fabricated real-world patient data in the final demo.

---

# 25. EVON OUTPUT CONTRACT

Evon should preferably return structured JSON where possible.

Example:

{
  "intent": "adverse_event_report",
  "product": "Lantus SoloStar",
  "events": ["rash", "itching"],
  "onset": "8 hours after administration",
  "seriousness": "unknown",
  "outcome": "unknown",
  "missing_information": ["age", "dose", "outcome"],
  "next_question": "What is your date of birth?",
  "escalation": false
}

Python validates the output before using it.

Never trust free-form LLM output blindly.

---

# 26. MODEL RESPONSIBILITY SPLIT

### Prisma
Speech → text.

### Evon
Reasoning/orchestration support:
- intent
- extraction
- conversational interpretation
- question generation
- summarization

### Timbre
Text → speech.

### Python/FastAPI
System control:
- state machine
- safety rules
- validation
- RAG retrieval
- case creation
- persistence
- logging
- API orchestration

### FAISS/RAG
Reference retrieval.

### Frontend
User interaction and visualization.

---

# 27. TECH STACK

Core:
- Gnani Prisma v2.5
- Gnani Evon v3.3
- Gnani Timbre v2.5

Backend:
- Python
- FastAPI

Frontend:
- Lovable for polished UI
- Streamlit only if needed for quick internal prototyping

AI/RAG:
- PyMuPDF
- embeddings
- FAISS

Development:
- Cursor
- Git
- GitHub
- Postman
- pytest

Deployment:
- Render for FastAPI backend
- appropriate frontend hosting for Lovable output

Presentation:
- Canva
- OBS if screen recording is required

Avoid unnecessary complexity:
- no n8n for the core build
- no Figma dependency
- no complex cloud architecture unless genuinely required

---

# 28. REPOSITORY STRUCTURE

Recommended:

medvaani/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   ├── models/
│   ├── services/
│   │   ├── prisma.py
│   │   ├── evon.py
│   │   ├── timbre.py
│   │   ├── safety.py
│   │   ├── rag.py
│   │   └── case_manager.py
│   ├── workflows/
│   │   ├── ae_workflow.py
│   │   └── mi_workflow.py
│   └── utils/
│
├── data/
│   ├── reference/
│   └── vectorstore/
│
├── tests/
│
├── frontend/
│
├── docs/
│
├── .env.example
├── requirements.txt
├── README.md
└── MASTER_CONTEXT.md

Exact structure may change during implementation, but the responsibility separation should remain.

---

# 29. DEVELOPMENT ORDER

Do NOT build the entire polished product at once.

### Phase 1 — API validation
Test:
1. Prisma
2. Evon
3. Timbre

Confirm each works independently.

### Phase 2 — basic voice loop
Audio
→ Prisma
→ Evon
→ Timbre
→ audio

### Phase 3 — AE state machine
Build deterministic Python workflow.

### Phase 4 — safety layer
Add:
- MI safety rules
- AE urgent-event interruption
- escalation flags

### Phase 5 — RAG
Add Lantus reference retrieval.

### Phase 6 — chat UI
Build the text conversational experience.

### Phase 7 — voice UI
Add microphone, waveform, transcript confirmation and audio response.

### Phase 8 — reviewer dashboard
Display structured AE cases.

### Phase 9 — integration
Connect UI + API + models + RAG + case storage.

### Phase 10 — testing
Test:
- normal AE
- urgent AE
- incomplete information
- incorrect transcription
- Hinglish
- Indian-language speech
- medical terms
- unsupported MI questions
- personalized treatment requests
- hallucination resistance

### Phase 11 — deployment
Deploy backend and frontend.

### Phase 12 — demo/presentation
Create:
- 60-second demo
- architecture diagram
- product story
- case study
- GitHub README

---

# 30. COMPETITION DEMO FLOW

Recommended 60-second story:

1. Open Lantus SoloStar MedVaani page.
2. Choose "Report a Side Effect."
3. Speak naturally in Hinglish/Indian language.
4. Prisma transcribes.
5. Evon identifies AE intent and extracts event details.
6. Safety layer checks the message.
7. MedVaani asks the next required question.
8. User provides essential details.
9. System generates structured AE case.
10. Show reviewer dashboard.
11. Show how the same experience can be accessed through voice or chat.

Hero statement:

> **From a patient's voice to a structured safety report — without turning the conversation into a complicated form.**

---

# 31. DEMO SCENARIOS

## Scenario A — Normal AE

User:
> I took Lantus yesterday and since this morning I have a headache.

Expected:
- detect AE
- identify Lantus as suspect product
- identify headache
- ask onset/timing
- collect required identity/contact information
- capture consent
- continue structured report
- no diagnosis
- no causal conclusion

## Scenario B — Urgent safety event

User:
> After taking Lantus I'm having severe difficulty breathing.

Expected:
- safety interrupt
- urgent medical attention guidance
- high-priority flag
- pause normal questionnaire
- preserve case information
- human review

## Scenario C — Personalized treatment request

User:
> My sugar is 250. Should I increase my Lantus dose?

Expected:
- refuse personalized dosing guidance
- provide bounded product-information option
- recommend speaking with healthcare professional

## Scenario D — Multilingual / code-switched

User:
> Lantus injection eduthathinu shesham enikku body full rash vannu.

Expected:
- Prisma transcription
- Evon intent/event extraction
- conversational response in supported language
- same AE state machine

---

# 32. TESTING MATRIX

Test at minimum:

### Language
- English
- Hindi
- Hinglish
- Malayalam
- English + Indian-language switching

### Speech
- clear audio
- accent variation
- background noise
- low-volume speech
- medical terminology
- names
- numbers
- dates

### Safety
- urgent symptom
- treatment request
- diagnosis request
- unsupported question
- missing information
- ambiguous event

### Data quality
- duplicate information
- corrected information
- incomplete information
- user refuses optional field
- user changes answer
- user stops midway

### RAG
- answer explicitly in reference
- answer partially in reference
- answer absent from reference
- misleading question
- personalized medical question

---

# 33. SUCCESS METRICS

Do not fabricate results.

Potential evaluation metrics:

### Voice
- transcription accuracy
- medical-term transcription accuracy
- language/code-switch handling

### AE workflow
- intent classification accuracy
- field extraction accuracy
- required-field completion rate
- incorrect-question rate
- case-generation success rate

### Safety
- unsafe-response rate
- urgent-event detection rate
- false escalation rate
- unsupported-advice rate

### UX
- time to complete a report
- number of conversational turns
- abandonment rate
- user correction rate

### System
- end-to-end latency
- API cost per interaction
- retrieval relevance

Only publish measured values.

---

# 34. EVALUATION PRINCIPLE

The strongest portfolio story is not:

> "I built a chatbot."

It is:

> "I designed a safety-bounded conversational workflow that converts unstructured patient voice into structured pharmacovigilance information while keeping deterministic controls around an LLM."

Show:
- why the problem exists
- why voice matters
- why Indian-language/code-switching matters
- how AI is used
- where AI is deliberately NOT trusted
- how safety is enforced
- how the output becomes actionable

---

# 35. HUMAN-IN-THE-LOOP PRINCIPLE

MedVaani is an AI-assisted intake and information system.

Human reviewers remain responsible for:
- pharmacovigilance review
- medical assessment
- case follow-up
- causality assessment
- regulatory submission decisions
- clinical decisions

The AI should prepare and structure information, not replace qualified professionals.

---

# 36. HALLUCINATION PREVENTION RULES

Any AI assistant working on this project must:

1. Treat this document as the canonical product context.
2. Do not invent product integrations.
3. Do not invent regulatory requirements.
4. Do not invent API capabilities.
5. Do not invent Gnani model capabilities.
6. Do not invent evaluation results.
7. Do not invent patient data.
8. Do not change the core product definition without explicit approval.
9. Do not turn MedVaani into an AI doctor.
10. Do not remove the safety layer.
11. Do not remove human review from serious AE handling.
12. Do not replace the AE workflow with a generic chatbot.
13. Do not introduce unnecessary tools merely because they are available.
14. If a technical capability is uncertain, ask or verify rather than guessing.
15. If a product/regulatory claim is uncertain, verify against an authoritative source.

---

# 37. CHANGE CONTROL

Before making a major architectural/product change, check:

### Does this change:
- alter the product purpose?
- alter the AE workflow?
- alter mandatory fields?
- remove a safety guardrail?
- change the model responsibility split?
- introduce a new external integration?
- make a regulatory claim?
- introduce real patient data?
- claim real pharma partnership?

If YES:
pause and explicitly flag the change before implementing it.

Minor implementation decisions can be made autonomously when they do not change product behavior or safety.

---

# 38. CURRENT UI DIRECTION

Two primary AE interfaces are approved concept directions:

## A. Chat / Text Input

Left navigation:
- New Chat
- Get Product Information
- Report a Side Effect
- Previously Reported Adverse Events

Main:
- Reporting a Side Effect banner
- safety disclaimer
- conversational messages
- patient responses
- progress through required questions
- text input
- microphone shortcut
- emergency-help card

## B. Voice Input

Same overall shell.

Main:
- Reporting a Side Effect banner
- user voice message
- MedVaani response
- large listening microphone
- waveform
- transcription confirmation
- retry
- stop recording
- switch to keyboard

Both should look like the same product, not two unrelated applications.

---

# 39. UI COPY PRINCIPLES

Use calm, human language.

Prefer:
> "I can help you report this."

Avoid:
> "Initiating pharmacovigilance case creation protocol."

Prefer:
> "What happened?"

Avoid:
> "Please input adverse-event narrative."

Prefer:
> "I heard: ... Is that correct?"

Avoid:
> "Speech-to-text confidence threshold exceeded."

Technical information can be exposed in a separate transparency/reviewer panel.

---

# 40. AI TRANSPARENCY

The final product may include an expandable "How MedVaani handled this" panel.

Example:

Speech recognized
→ Prisma

Intent and information extracted
→ Evon

Safety check
→ Safety layer

Product information retrieved
→ Lantus reference

Voice response generated
→ Timbre

This is useful for the competition and portfolio because it makes the architecture understandable without exposing internal chain-of-thought.

Never expose hidden reasoning or chain-of-thought.

---

# 41. FINAL PRODUCT ARCHITECTURE

                 ┌─────────────────────┐
                 │      USER           │
                 │ Text / Voice        │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   Prisma v2.5       │
                 │ Speech → Text       │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   FastAPI           │
                 │ Orchestration       │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   Evon v3.3         │
                 │ Intent / Extraction │
                 └──────────┬──────────┘
                            │
             ┌──────────────▼──────────────┐
             │   ALWAYS-ACTIVE SAFETY     │
             │   Guardrails + Escalation  │
             └──────────────┬──────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
           ┌──────▼──────┐     ┌──────▼──────┐
           │ MI Workflow │     │ AE Workflow │
           └──────┬──────┘     └──────┬──────┘
                  │                   │
           ┌──────▼──────┐     ┌──────▼──────┐
           │ RAG / FAISS │     │ Case State  │
           │ Lantus Ref. │     │ + Validation│
           └──────┬──────┘     └──────┬──────┘
                  │                   │
                  └─────────┬─────────┘
                            │
                   ┌────────▼────────┐
                   │ Response Layer  │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Timbre v2.5     │
                   │ Text → Speech   │
                   └─────────────────┘

---

# 42. DELIVERY CHECKLIST

Before declaring MedVaani complete:

### Product
- [ ] Product identity consistent
- [ ] Lantus SoloStar demo context consistent
- [ ] AE is hero workflow
- [ ] MI is secondary workflow
- [ ] Text and voice supported

### AI
- [ ] Prisma integrated
- [ ] Evon integrated
- [ ] Timbre integrated
- [ ] Structured Evon output
- [ ] RAG integrated
- [ ] Hallucination controls

### Safety
- [ ] Safety guardrail always active
- [ ] MI treatment-request handling
- [ ] AE urgent-event interrupt
- [ ] Human review path
- [ ] No diagnosis
- [ ] No autonomous treatment recommendation

### UX
- [ ] High-fidelity landing page
- [ ] Chat AE interface
- [ ] Voice AE interface
- [ ] Previously Reported Adverse Events
- [ ] Reviewer dashboard
- [ ] Clear disclaimer
- [ ] Mobile/responsive behavior

### Testing
- [ ] English
- [ ] Hindi/Hinglish
- [ ] Malayalam
- [ ] noisy audio
- [ ] medical terms
- [ ] safety scenarios
- [ ] RAG unsupported questions
- [ ] incomplete cases

### Portfolio
- [ ] GitHub repository
- [ ] README
- [ ] architecture diagram
- [ ] product case study
- [ ] 60-second demo
- [ ] evaluation results
- [ ] screenshots
- [ ] limitations/future work

---

# 43. FUTURE FEATURES — DO NOT BUILD BY DEFAULT

Possible future directions:
- telephonic IVR
- real pharma safety-system integration
- CRM integration
- multilingual expansion
- human reviewer collaboration
- analytics dashboard
- automated follow-up requests
- configurable product libraries
- enterprise authentication

These are future roadmap items unless explicitly promoted into the active build.

Do not add them simply to make the project sound larger.

---

# 44. PROJECT NORTH STAR

The project should demonstrate:

> **Safe conversational AI for pharmaceutical information and pharmacovigilance intake — designed for real-world Indian voice interactions.**

The most important design principle is:

> **Use AI where natural language understanding helps. Use deterministic software where safety, state, validation and control matter.**

When making future decisions, optimize for:
1. safety
2. correctness
3. demo clarity
4. user experience
5. technical credibility
6. competition relevance
7. portfolio value

Do not optimize for feature count.

---

# 45. INSTRUCTION FOR FUTURE AI ASSISTANTS

Before answering any MedVaani-related question, first identify:

1. Which workflow?
   - MI
   - AE
   - Shared platform
   - Reviewer

2. Which layer?
   - UI
   - Backend
   - Prisma
   - Evon
   - Timbre
   - RAG
   - Safety
   - Data/case management

3. Is the requested change consistent with the canonical product definition?

4. Does it affect safety or regulatory claims?

5. Does it require verification?

If uncertain:
- do not guess
- state the uncertainty
- verify authoritative information when appropriate
- keep the existing architecture intact

---

# 46. CURRENT BUILD STATUS

Concept and product direction:
**Defined**

Brand:
**Defined — MedVaani**

Primary demo medication:
**Defined — Lantus® SoloStar®**

AE:
**Hero workflow**

MI:
**Secondary workflow**

Chat AE UI:
**Concept approved**

Voice AE UI:
**Concept approved**

Navigation:
**Includes "Previously Reported Adverse Events"**

Safety architecture:
**Defined**

RAG:
**Planned**

Backend:
**FastAPI planned**

Frontend:
**Lovable planned**

Gnani models:
**Prisma + Evon + Timbre planned**

Deployment:
**Render / suitable frontend hosting planned**

Competition demo:
**Planned**

---

# END OF MASTER CONTEXT

This document is the canonical reference for MedVaani. Any future AI-assisted build, UI generation, code generation, prompt creation, documentation, testing or presentation work should preserve this context unless the project owner explicitly changes it.
