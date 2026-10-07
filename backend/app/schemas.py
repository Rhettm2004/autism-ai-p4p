from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.:-]+$')]
UnitFloat = Annotated[float, Field(ge=0, le=1)]
ModelAlias = Literal['mistral', 'llama']
Route = Literal['safety_deflect', 'misinformation_correction', 'screening_guidance',
                'result_explanation', 'referral', 'general_knowledge', 'caregiver_support']
Stage = Literal['welcome', 'toddlerCheck', 'respondentDetails', 'backgroundQuestions',
                'behaviouralQuestions', 'review', 'disclaimer', 'result', 'validation', 'report']
Questionnaire = Literal['qchat10', 'aq10Child', 'aq10Adolescent', 'aq10Adult']

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class HistoryMessage(StrictModel):
    role: Literal['user', 'assistant']
    content: Text

class CurrentQuestion(StrictModel):
    id: Identifier
    number: int = Field(ge=1, le=10)
    text: Text

class QuestionnaireQuestion(StrictModel):
    id: Identifier
    number: int = Field(ge=1, le=10)
    text: Text

class PredictionResult(StrictModel):
    traits_detected: bool
    similarity_percentage: float = Field(ge=0, le=100)
    is_mock: bool
    disagreement: float | None = Field(default=None, ge=0, le=1)
    confidence_pi: float | None = Field(default=None, ge=0, le=1)
    tuned_threshold: float | None = Field(default=None, ge=0, le=1)
    per_module_raw_probability: dict[str, UnitFloat] = Field(default_factory=dict)
    per_module_calibrated_probability: dict[str, UnitFloat] = Field(default_factory=dict)
    agreement_scores: dict[str, UnitFloat] = Field(default_factory=dict)

class RespondentDetails(StrictModel):
    is_toddler: bool | None = None
    age: int | None = Field(default=None, ge=0, le=120)
    age_unit: Literal['months', 'years'] | None = None
    gender: Text | None = None
    ethnicity: Text | None = None

class BackgroundDetails(StrictModel):
    jaundice: bool | None = None
    family_autism_history: bool | None = None
    completed_by: Text | None = None

class ScreeningContext(StrictModel):
    revision: int = Field(default=0, ge=0)
    stage: Stage
    screening_active: bool
    questionnaire: Questionnaire | None = None
    current_question: CurrentQuestion | None = None
    questionnaire_questions: list[QuestionnaireQuestion] = Field(default_factory=list, max_length=10)
    prediction_result: PredictionResult | None = None
    respondent_details: RespondentDetails = Field(default_factory=RespondentDetails)
    background_details: BackgroundDetails = Field(default_factory=BackgroundDetails)

    @model_validator(mode='after')
    def consistent(self):
        if self.current_question is not None and (
            self.stage != 'behaviouralQuestions' or self.questionnaire is None
        ):
            raise ValueError('Current question requires an active questionnaire stage')
        if self.questionnaire_questions and self.questionnaire is None:
            raise ValueError('Questionnaire questions require a selected questionnaire')
        if len({question.number for question in self.questionnaire_questions}) != len(self.questionnaire_questions):
            raise ValueError('Questionnaire question numbers must be unique')
        return self

class ChatOptions(StrictModel):
    router: bool = True
    rag: bool = True
    cite: bool = True
    concise: bool = True

class ChatRequest(StrictModel):
    api_version: Literal[1] = 1
    request_id: Identifier
    session_id: Identifier
    message: Text
    history: list[HistoryMessage] = Field(default_factory=list, max_length=60)
    model: ModelAlias = 'mistral'
    options: ChatOptions = Field(default_factory=ChatOptions)
    screening_context: ScreeningContext

class Source(StrictModel):
    number: int = Field(ge=1)
    title: str
    url: str
    authority: str
    low_authority: bool
    passage_ids: list[str]
    cited: bool

class ChatMetadata(StrictModel):
    runtime_profile: str
    prompt_profile: str
    router_config: str
    route_decided_by: str
    rag_used: bool
    corpus_sha256: str
    prompts_sha256: str
    training_sha256: str
    application_prompt_version: int | None = None
    model_identity: str
    finish_reason: str | None = None
    invalid_citations: list[int] = Field(default_factory=list)

class CommandResult(StrictModel):
    name: str
    arg: str | int | None = None
    executed_question: str | None = None

class StartScreeningAction(StrictModel):
    type: Literal['start_screening']
    expected_context_revision: int = Field(ge=0)

class ChatResponse(StrictModel):
    api_version: Literal[1] = 1
    request_id: str
    session_id: str
    context_revision: int
    response: str
    route: Route
    model: ModelAlias
    sources: list[Source]
    options: ChatOptions
    command: CommandResult | None = None
    action: StartScreeningAction | None = None
    metadata: ChatMetadata

class EaipFeatures(StrictModel):
    q1: Literal[0, 1] = Field(alias='Q1')
    q2: Literal[0, 1] = Field(alias='Q2')
    q3: Literal[0, 1] = Field(alias='Q3')
    q4: Literal[0, 1] = Field(alias='Q4')
    q5: Literal[0, 1] = Field(alias='Q5')
    q6: Literal[0, 1] = Field(alias='Q6')
    q7: Literal[0, 1] = Field(alias='Q7')
    q8: Literal[0, 1] = Field(alias='Q8')
    q9: Literal[0, 1] = Field(alias='Q9')
    q10: Literal[0, 1] = Field(alias='Q10')
    age: int = Field(alias='Age', ge=1, le=120)
    sex: Literal['m', 'f'] = Field(alias='Sex')
    ethnicity: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)] = Field(alias='Ethnicity')
    jauntice: Literal['yes', 'no'] = Field(alias='Jauntice')
    family_asd_history: Literal['yes', 'no'] = Field(alias='FamilyASDHistory')
    autism_age_category: Literal['chat', 'child', 'adolescent', 'adult'] = Field(alias='AutismAgeCategory')

class ScreeningPredictionRequest(StrictModel):
    api_version: Literal[1] = 1
    request_id: Identifier
    session_id: Identifier
    features: EaipFeatures

class EaipModuleValues(StrictModel):
    M1_screening: float | None = Field(default=None, ge=0, le=1)
    M2_ASSL: float | None = Field(default=None, ge=0, le=1)
    M3_cluster: float | None = Field(default=None, ge=0, le=1)

class EaipThresholds(StrictModel):
    eaip: float = Field(ge=0, le=1)
    darv_fixed: float = Field(ge=0, le=1)
    darv_tuned: float = Field(ge=0, le=1)

class ScreeningPredictionResponse(StrictModel):
    api_version: Literal[1] = 1
    request_id: str
    session_id: str
    model: Literal['eaip-darv'] = 'eaip-darv'
    eaip_probability: float = Field(ge=0, le=1)
    darv_probability: float = Field(ge=0, le=1)
    classification_eaip: Literal[0, 1]
    classification_darv_fixed: Literal[0, 1]
    classification_darv_tuned: Literal[0, 1]
    disagreement: float = Field(ge=0, le=1)
    confidence_pi: float = Field(ge=0, le=1)
    per_module_raw_probability: EaipModuleValues
    per_module_calibrated_probability: EaipModuleValues
    agreement_scores: EaipModuleValues
    thresholds_used: EaipThresholds
