# Autism AI: EAIP-DARV integration and interface development record

**Prepared:** 5 October 2026 (Pacific/Auckland)  
**Scope:** Repository review, explanations, decisions, implementation, research, and verification undertaken in this chat.  
**Purpose:** Supporting material for the project research report. This document distinguishes functionality already present when the review began from changes made during the chat. It records implementation progress rather than evidence of clinical effectiveness.

## 1. Project context and requirements

The project combines a Flutter screening interface with a conversational assistant and a Python classification service. The supplied `EAIP ASD project.zip` contains trained model artifacts, preprocessing configurations, calibration artifacts, prediction code, and an HTTP API.

Rabia's correspondence established the following integration requirements:

- Collect the ten behavioural responses and demographic/background fields in a structured form.
- Use the classifier's live `GET /schema` endpoint to check expected feature keys.
- Pass the structured feature record to `POST /predict`.
- Allow the deployment package to apply its saved preprocessing.
- Keep Llama/Mistral responsible for question explanations and side Q&A; classifier inputs must come directly from Flutter controls.
- Keep screening data in local session storage while university storage permission remains unresolved.
- Improve the user interface and clarify how screening responses and chat history are stored.

The actual supplied API expects a request shaped as `{"features": {...}}`. The contents of `features` are a flat field-value dictionary. Although the email described a contract containing types and expected values, the supplied `/schema` implementation returns only `expected_raw_columns`. Types, category meanings, age units, and answer encoding therefore require additional confirmation.

## 2. Initial repository review

The first activity was a read-only review of the current application and ZIP. An initial EAIP integration was already present in the working copy. It was not created from scratch in this chat.

Existing functionality included:

- A chat-first Flutter interface with inline structured screening cards.
- A controller managing screening stages, answer review, editing, results, and session restoration.
- Age-specific ten-question banks: Q-CHAT-10 for toddlers and AQ-10 variants for children, adolescents, and adults.
- Deterministic conversion from selected answers to binary model fields.
- A Flutter EAIP prediction client and a typed FastAPI screening endpoint.
- An isolated model service loading the supplied three-module bundle.
- Local persistence of screening state, chat messages, and results.
- A PDF report generator.
- Explicit mock providers for interface development and tests.

The model runtime already contained compatibility handling for Windows-authored artifact paths and Keras weight loading on macOS. Those compatibility loaders were inspected but not newly implemented in this chat.

## 3. Application architecture and workflow

### 3.1 Screening workflow

The controller supports this sequence:

1. Welcome and agreement to begin screening.
2. Selection of the age pathway.
3. Respondent details.
4. Background questions.
5. Ten behavioural questions.
6. Review and editing.
7. Disclaimer acknowledgement.
8. Model prediction and result.

The final interface makes report access immediate after prediction. Research validation is no longer a required step for accessing the report. Validation controller functionality remains in the repository, but the final result card no longer displays its optional validation action.

### 3.2 Separate processing paths

```text
Conversational path
Flutter chat -> FastAPI /chat -> research router -> RAG retrieval
             -> Llama/Mistral -> assistant reply and sources

Screening path
Flutter form fields and answer controls -> deterministic answer encoding
    -> GET /screening/schema
    -> POST /screening/predict
    -> backend checks live EAIP schema
    -> EAIP service POST /predict
    -> preprocessing, three model predictions, calibration, EAIP and DARV
    -> Flutter result, local submission record, input viewer, user report
```

The conversational backend reuses a research router, TF-IDF retrieval, prompts, source handling, and a llama.cpp adapter. These components were reviewed as context; their research methodology was not changed during this chat.

An assistant reply can explain questions and provide next-step guidance. It does not extract age, demographics, or behavioural answers from free text to populate the classifier input.

### 3.3 Technologies

| Component | Technology and responsibility |
| --- | --- |
| User interface | Flutter/Dart; structured controls, conversation, screening stages, dialogs |
| App API | Python FastAPI/Pydantic; typed requests, validation, screening proxy |
| HTTP transport | Dart `http` and Python `httpx` |
| Conversational generation | Local Llama/Mistral through llama.cpp |
| Retrieval | Existing TF-IDF RAG pipeline and research router |
| Classification | TensorFlow/Keras and PyTorch CNN artifacts |
| Saved transformations | JSON preprocessing configuration and joblib preprocessors/calibrators |
| Local persistence | SharedPreferences session JSON |
| PDF generation | Dart `pdf` package |
| Verification | Flutter analyzer/tests, Python pytest, rendered PDF inspection |

## 4. How the supplied EAIP-DARV system works

All three modules run for each completed screening in the integrated service. Age selects the questionnaire, not a particular model module.

| Module | Deployed predictor |
| --- | --- |
| M1 screening | TensorFlow/Keras screening CNN |
| M2 ASSL | PyTorch CNN produced by the ASSL training pipeline |
| M3 cluster | TensorFlow/Keras final CNN from the clustering-based training pipeline |

The deployment loads saved predictors rather than rerunning training, semi-supervised learning, or clustering for each user.

### 4.1 Prediction stages

1. Each module applies its saved preprocessing and produces a raw probability.
2. Saved calibration transforms each module's probability. Calibrators are loaded rather than fitted during screening.
3. Agreement scores measure how closely each calibrated prediction matches the other modules.
4. EAIP produces an agreement-weighted combined probability.
5. DARV adjusts the combined probability toward 0.5 when the modules disagree.
6. Saved thresholds determine binary classifications.

For calibrated module probabilities `p_k`, the supplied implementation computes:

```text
A_k = average over other modules of (1 - abs(p_k - p_j))
EAIP = sum(A_k * p_k) / sum(A_k)
D = average pairwise absolute difference between module probabilities
pi = exp(-lambda * D)
DARV = pi * EAIP + (1 - pi) * 0.5
```

The supplied configuration uses `lambda = 0.5`, an EAIP threshold of `0.5`, a fixed DARV threshold of `0.5`, and a tuned DARV threshold of approximately `0.52`. The application uses the returned tuned DARV classification as its main flag and the DARV probability as its main score.

`confidence_pi` is a DARV agreement weight, not an accuracy estimate or certainty that a person is autistic. Displaying it as a percentage does not change its meaning. The final implementation keeps this and other technical measures inside the inspection dialog.

## 5. Model inputs and outputs

### 5.1 Current input fields

| Field | Current application representation |
| --- | --- |
| `Q1`–`Q10` | Ten binary, question-specific encoded responses |
| `Age` | Integer age; toddler pathway currently uses months, other pathways use years |
| `Sex` | `m` or `f` |
| `Ethnicity` | Selected category |
| `Jauntice` | `yes` or `no`; model's original spelling retained |
| `FamilyASDHistory` | `yes` or `no` |
| `AutismAgeCategory` | `chat`, `child`, `adolescent`, or `adult` |

Chat text, who completed the screening, and research-validation responses are not classifier features. They may be stored locally or included in a user summary where relevant.

Age affects both questionnaire selection and prediction inputs. Its precise influence on a prediction was not measured in this chat. Mixed age units and category conventions still require verification against training records.

### 5.2 Output retained by the application

The full model response is now retained locally, including:

- Raw and calibrated probabilities for each module.
- Per-module agreement scores.
- EAIP and DARV probabilities.
- EAIP, fixed-DARV, and tuned-DARV classifications.
- Disagreement and DARV agreement weight.
- Thresholds used.
- Response identifiers and other returned metadata.

The established Flutter result fields remain available for existing consumers. The captured `modelResponse` retains the complete response for testing and inspection.

## 6. Answer encoding and scoring research

A user test selected “Definitely Agree” for every adult question and produced a raised model flag. The user questioned why some encoded values were zero and suggested that all questions except question 10 should become one.

The published Autism Research Centre adult AQ-10 scoring key was checked. It specifies:

- Agreement scores one on adult items **1, 7, 8, and 10**.
- Disagreement scores one on adult items **2, 3, 4, 5, 6, and 9**.

Consequently, choosing “Definitely Agree” for every adult question produces:

```text
Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10
 1  0  0  0  0  0  1  1  0   1
```

The existing adult encoder matched the published key and was deliberately retained. Agreement is not always the scored response because questions describe different abilities and difficulties. Slight and definite agreement have the same binary value under this key; the same applies to the two disagreement options.

The child and adolescent keys were also checked:

| Questionnaire | Items scoring one for agreement |
| --- | --- |
| AQ-10 adult | 1, 7, 8, 10 |
| AQ-10 child | 1, 5, 7, 10 |
| AQ-10 adolescent | 1, 5, 8, 10 |

Regression coverage now explicitly checks all four adult response options against the expected item vector. Source references were added to the encoder comments.

**Important distinction:** verifying the conventional questionnaire key does not prove that Rabia's training data used the same feature encoding. The observed model score of 53.4% cannot be judged correct or incorrect from the questionnaire key alone. The model also uses background fields, calibration, combination, and a tuned threshold. It is not a conventional questionnaire sum.

The published child questionnaire describes ages 4–11, whereas the current app routes age 3 to the child pathway. This boundary was not changed in this chat and should be reviewed with the model owner as part of age-contract validation.

## 7. Preprocessing explanation and implementation boundary

Answer encoding and preprocessing are separate activities:

- **Answer encoding:** Flutter maps a selected answer to its question-specific binary feature.
- **Preprocessing:** each saved model preprocessor transforms the feature record into the representation expected by that model.

Preprocessing can include missing-value handling, numeric scaling, categorical encoding, feature ordering, and tensor reshaping. Saved training statistics are reused; the app does not fit them from user submissions.

The pre-existing integrated service additionally converts sex and yes/no fields to numeric values for Module 1. It passes categorical values to Modules 2 and 3. These conversions were documented in the inspection dialog, but their fidelity to the original model environment was not independently established in this chat.

## 8. Implementation completed during this chat

### 8.1 Live schema validation

- Added the app-backend `GET /screening/schema` endpoint.
- Added an EAIP client method forwarding the model service's schema.
- Flutter retrieves expected keys before sending a prediction.
- Flutter builds the submitted feature dictionary using those keys and rejects unsupported or null required values.
- The backend checks the live key set before forwarding predictions.
- Invalid, unavailable, or incompatible schemas produce errors rather than silent feature imputation or mock fallback.

This is compatibility validation, not automatic support for arbitrary future features. The backend still has a typed feature contract. Schema changes requiring new fields or a different contract need an application update. The schema itself does not document values and units.

### 8.2 Captured submission and response

The result model now stores:

- A submission record containing exact sent features.
- Original behavioural question wording and selected responses.
- A snapshot of respondent/background/session data.
- Submission timestamp.
- The full returned model response.

The captured record is saved with the local session. Reports use the submission-time snapshot for respondent fields and answers, so later state changes do not silently rewrite the basis of the result. Older and mock results explicitly indicate when no captured submission is available.

### 8.3 Interface development and refinement

The first implementation added the three result actions and an expandable calculation panel. A subsequent user review identified awkward navigation and excessive technical detail in the report. The final design is:

| Element | Final behaviour |
| --- | --- |
| Main result | Plain-language threshold outcome and model screening score |
| View EAIP inputs | Opens a separate closable dialog |
| View / download report | Opens a separate closable user-report dialog |
| Next steps | Requests contextual explanation and next-step guidance in chat |
| Start new screening | Available in the header; text on wider screens and an icon with tooltip on narrower screens |

Opening and closing either dialog leaves the completed result and its three buttons in place. The report no longer replaces the result inside the chat. The old “Continue Conversation” action and report-level restart action were removed. Existing restored report-stage sessions display the same result actions.

The input dialog shows original answers beside encoded values, background features, the exact classifier request, an explanation of question-specific scoring, and expandable technical calculations/full response.

Raised results use an informational icon and warm styling; result wording no longer presents the model name as the main user-facing message. Mock results remain explicitly labelled.

### 8.4 User report and PDF

The report was initially expanded with technical model calculations and an input appendix for development. Following user feedback, these were removed from the final user report.

The final user report contains:

- Screening outcome and a plain-language explanation.
- Screening score with an explanation that it is not percentage certainty of autism.
- Respondent and background details.
- Original questions and selected answers.
- Screening disclaimer.

It does not expose raw JSON, encoded model inputs, module predictions, agreement calculations, or an integration appendix. Technical information remains available through the separate inspection action. An unanswered research-validation section is not included in the user PDF.

## 9. Local storage and chat history

Screening state, chat messages, submission records, and results remain in the existing SharedPreferences-backed local session store. No cloud or university storage was added.

This is a restorable session on the current device/browser rather than an authenticated multi-user conversation archive. Local persistence should not be described as encryption or as a complete production data-management solution; those capabilities were not implemented in this chat.

Starting a new screening uses the existing restart confirmation and reset behaviour, which clears the current screening and chat history. This does not implement preservation of multiple previous sessions.

## 10. Verification and observed results

| Check | Outcome recorded in this chat |
| --- | --- |
| Final Flutter analysis | No issues found |
| Final permanent Flutter tests | 58 passed |
| Backend application tests after schema changes | 47 passed; one dependency deprecation warning |
| Whitespace/diff checks | Passed |
| Generated PDF | Rendered and visually inspected; final user report was two pages for the sample |
| Popup navigation test | Inputs and report close back to the same result; actions remain available |
| Adult scoring regression | Published forward/reverse scoring checked across all four answer options |
| Persistence check | Captured features and response survive serialization/restoration |
| Schema-change check | Unsupported required field prevents model POST |

An initial popup test failed because the test clicked before a scroll animation completed. The test was corrected to wait for the animation, then passed. A PDF section-heading page-break issue in the earlier technical version was identified by visual inspection and corrected.

The UI/PDF tests use sample or mocked outputs. They demonstrate software behaviour, not classifier accuracy. Earlier repository documentation reported live model checks, but those were pre-existing records and were not independently rerun during this chat.

The user observed a live result of 53.4% with all adult answers set to agreement. This is a user-reported test observation, not a benchmark or verified reference prediction. No new live classifier prediction or model-quality evaluation was performed by the agent during this chat.

## 11. Remaining work and research limitations

Before making claims about model correctness, research equivalence, or clinical usefulness:

1. Obtain reference input records and expected per-module/final outputs from Rabia.
2. Confirm question wording, question order, binary scoring, category values, missing-value policy, and age units against training data.
3. Review the child-pathway age boundary and all supported age ranges.
4. Compare the integrated compatibility loaders and Module 1 conversions with the original deployment environment.
5. Confirm that tuned DARV is the intended primary output and document the threshold/calibration provenance.
6. Test complete live Flutter submissions against reference results.
7. Resolve storage permissions with Reza/IT before adding remote storage.
8. Define requirements for multiple local conversations, user identities, retention, and deletion if those become part of the scope.

The software changes make inputs inspectable and navigation clearer. They do not establish diagnostic validity, explain feature importance, or measure the clinical effect of age.

## 12. How to reproduce testing

From the project root:

```bash
./run_app.sh
```

Select Llama or Mistral for integrated testing. Use mock mode for interface-only testing. If the EAIP environment is not prepared:

```bash
./setup_eaip.sh
```

Complete a screening, inspect the submitted feature values, close the dialog, open/download the user report, close it, and request next steps. Refresh and restore the session to check persistence. Confirm the header restart control remains available.

Automated checks:

```bash
flutter analyze
flutter test
```

```bash
cd backend
.venv/bin/python -m pytest app_tests -q
```

Compare live results with model-owner reference outputs to test prediction correctness; automated interface tests alone do not do that.

## 13. Main implementation files

| File | Relevant responsibility |
| --- | --- |
| `lib/state/screening_controller.dart` | Screening workflow and local state |
| `lib/services/questionnaire_response_encoder.dart` | Binary answer mapping and scoring source references |
| `lib/models/eaip_model_input.dart` | Structured feature preparation |
| `lib/services/eaip_screening_prediction_service.dart` | Schema fetch, submission capture, prediction and response retention |
| `lib/models/screening_models.dart` | Result/submission fields |
| `lib/models/screening_session.dart` | Local result serialization/restoration |
| `lib/screens/screening_page.dart` | Closable input/report dialogs |
| `lib/widgets/stage_cards.dart` | Result actions, user report and technical calculation view |
| `lib/widgets/app_header.dart` | Persistent restart control |
| `lib/services/report_service.dart` | Snapshot-based report data and user PDF generation |
| `backend/app/main.py` | Schema and prediction API routes |
| `backend/app/adapters/eaip_darv.py` | Model-service schema validation and prediction forwarding |
| `backend/eaip_service/app.py` | Pre-existing isolated model runtime |
| `test/result_popups_test.dart` | Dialog navigation regression |
| `test/eaip_result_actions_test.dart` | Stable result actions |
| `test/eaip_screening_prediction_service_test.dart` | Submission and response handling |
| `test/report_service_test.dart` | Encoding and report checks |
| `backend/app_tests/test_eaip_schema.py` | Schema incompatibility/error checks |

## 14. Sources

- Supplied local deployment package: `EAIP ASD project.zip`; inspected API, predictor, core mathematics, preprocessing configuration, module contracts, and calibration configuration.
- [Autism Research Centre: adult AQ-10 questionnaire and scoring key](https://docs.autismresearchcentre.com/tests/AQ10.pdf).
- [Autism Research Centre: child AQ-10 questionnaire and scoring key](https://docs.autismresearchcentre.com/tests/AQ10-Child.pdf).
- [Autism Research Centre: adolescent AQ-10 questionnaire and scoring key](https://docs.autismresearchcentre.com/tests/AQ10-Adolescent.pdf).
- Current application source and tests in this repository.

The scoring documents support questionnaire encoding rules. They are not evidence for the supplied EAIP model's predictive performance or training contract.
