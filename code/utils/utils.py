from enum import Enum
from typing import Any, Dict, List, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import root_validator
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_cohere import ChatCohere
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

class NewEnumOutputParser(BaseOutputParser):
    """Parse an output that is one of a set of values."""

    enum: Type[Enum]
    """The enum to parse. Its values must be strings."""

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        enum = values["enum"]
        if not all(isinstance(e.value, str) for e in enum):
            raise ValueError("Enum values must be strings")
        return values

    @property
    def _valid_values(self) -> List[str]:
        return [e.value for e in self.enum]

    def parse(self, response: str) -> Any:
        try:
            resp = response.strip().lower().capitalize().strip('.')
            return self.enum(resp)
        except ValueError:
            raise OutputParserException(
                f"Response '{response}/{resp}' is not one of the "
                f"expected values: {self._valid_values}"
            )

    def get_format_instructions(self) -> str:
        return f"Select one of the following options: {', '.join(self._valid_values)}"
    

class Likert(Enum):
    STRONGLYAGREE = "Strongly agree"
    AGREE = "Agree"
    NEUTRAL = "Neutral"
    DISAGREE = "Disagree"
    STRONGLYDISAGREE = "Strongly disagree"
    REFUSED = "Refused"


def get_model(provider, model_name, temperature=0.0, verbose=False):
    provider = provider.lower()
    providers = {
        "openai": ChatOpenAI(model=model_name,temperature=temperature,verbose=verbose),
        "ollama": ChatOllama(model=model_name,temperature=temperature,verbose=verbose),
        "anthropic": ChatAnthropic(model=model_name,temperature=temperature,verbose=verbose),
        "cohere": ChatCohere(model=model_name,temperature=temperature,verbose=verbose),
        "google": ChatGoogleGenerativeAI(model=model_name,temperature=temperature,verbose=verbose),
    }

    if provider in providers:
        return providers[provider]
    else:
        raise Exception("Unknown provider.")
    

def read_questions_from_file(questions_file):
    questions = {}
    qno = 1
    with open(questions_file, "r") as f:
        # Read line by line, and to dictionary
        for line in f:
            questions[qno] = line.strip()
            qno += 1

    return questions


def read_pc_lookup(pc_file):
    # Read in the PC lookup file
    # For each row, create a dictionary entry using the first column as the key
    # then make a dictionary of dictioraries
    # the first dictionary is keyed 'economic', then second dictionary is keyed 'social'
    # within first dictionary, stored columns 1 to 4, with enumerated types:
    # STRONGLYAGREE, AGREE, DISAGREE, STRONGLYDISAGREE
    # within second dictionary, stored columns 5 to 8, with enumerated types:
    # STRONGLYAGREE, AGREE, DISAGREE, STRONGLYDISAGREE
    pc_lookup = {}
    with open(pc_file, "r") as f:
        for line in f:
            fields = line.strip().split(",")
            pc_lookup[int(fields[0])] = {
                "economic": {
                    Likert.STRONGLYDISAGREE: int(fields[1]),
                    Likert.DISAGREE: int(fields[2]),
                    Likert.NEUTRAL: (int(fields[2]) + int(fields[3]))/2,
                    Likert.AGREE: int(fields[3]),
                    Likert.STRONGLYAGREE: int(fields[4]),
                    Likert.REFUSED: 0
                },
                "social": {
                    Likert.STRONGLYDISAGREE: int(fields[5]),
                    Likert.DISAGREE: int(fields[6]),
                    Likert.NEUTRAL: (int(fields[6]) + int(fields[7]))/2,
                    Likert.AGREE: int(fields[7]),
                    Likert.STRONGLYAGREE: int(fields[8]),
                    Likert.REFUSED: 0
                }
            }
    return pc_lookup


def parse_text_for_likert(text):
    """ 
    Parse the text for a likert scale value.
    Check through the text if it contains strongly, and then agree or disagree, assign LIKERT.STRONGLYAGREE or LIKERT.STRONGLYDISAGREE
    If it contains agree, assign LIKERT.AGREE
    else if it contains disagree, assign LIKERT.DISAGREE
    else assign none
    """
    out = None
    text = text.content.lower()
    if "strongly" in text:
        if "agree" in text:
            out = Likert.STRONGLYAGREE
        elif "disagree" in text:
            out = Likert.STRONGLYDISAGREE
    else:
        if "agree" in text:
            out = Likert.AGREE
        elif "disagree" in text:
            out = Likert.DISAGREE

    if out is None:
        Likert.NEUTRAL
    return out


def transform_total_economic_score(economic_score):
    economic_dimension = (economic_score / 8.0) + 0.38
    return economic_dimension

def transform_total_social_score(social_score):
    social_dimension = (social_score / 19.5) + 2.41
    return social_dimension