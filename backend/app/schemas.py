from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.:-]+$')]
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

class ClassicalResult(StrictModel):
    questionnaire: Questionnaire
    score: int = Field(ge=0, le=10)
    referral_threshold: int = Field(ge=0, le=10)
    threshold_met: bool

    @model_validator(mode='after')
    def consistent(self):
        if self.threshold_met != (self.score >= self.referral_threshold):
            raise ValueError('Inconsistent classical result')
        return self

class PredictionResult(StrictModel):
    traits_detected: bool
    similarity_percentage: float = Field(ge=0, le=100)
    is_mock: bool

class ScreeningContext(StrictModel):
    revision: int = Field(default=0, ge=0)
    stage: Stage
    screening_active: bool
    questionnaire: Questionnaire | None = None
    current_question: CurrentQuestion | None = None
    classical_result: ClassicalResult | None = None
    prediction_result: PredictionResult | None = None

    @model_validator(mode='after')
    def consistent(self):
        if self.current_question is not None and (
            self.stage != 'behaviouralQuestions' or self.questionnaire is None
        ):
            raise ValueError('Current question requires an active questionnaire stage')
        if self.classical_result and self.classical_result.questionnaire != self.questionnaire:
            raise ValueError('Result questionnaire does not match context')
        return self

class ChatRequest(StrictModel):
    api_version: Literal[1] = 1
    request_id: Identifier
    session_id: Identifier
    message: Text
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)
    model: ModelAlias = 'mistral'
    screening_context: ScreeningContext

class Source(StrictModel):
    number: int = Field(ge=1)
    title: str
    url: str
    authority: str
    low_authority: bool
    passage_ids: list[str]

class ChatMetadata(StrictModel):
    runtime_profile: str
    router_config: str
    route_decided_by: str
    rag_used: bool
    corpus_sha256: str
    prompts_sha256: str
    training_sha256: str
    application_prompt_version: int
    model_identity: str
    finish_reason: str | None = None

class ChatResponse(StrictModel):
    api_version: Literal[1] = 1
    request_id: str
    session_id: str
    context_revision: int
    response: str
    route: Route
    model: ModelAlias
    sources: list[Source]
    action: None = None
    metadata: ChatMetadata
