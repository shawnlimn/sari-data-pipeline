from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class DataMetadata(BaseModel):
    main_category: str = Field(..., description="Main category of the question")
    sub_category: Optional[List[str]] = Field(
        default=[], description="Sub-category or specific topic of the question"
    )
    data_source: str = Field(
        default="sari-source",
        description="Source of the dataset, typically a huggingface id",
    )
    cot_source: Optional[List[str]] = Field(
        default=[], description="List of Chain of Thought sources, e.g., DeepSeek-R1"
    )

    model_config = ConfigDict(extra="allow")


class DataEntry(BaseModel):
    uuid: str = Field(..., description="Unique identifier based on the question")
    question: str = Field(..., description="Clean question text")
    sft_data: List[Tuple[str, str]] = Field(
        ...,
        description="List of complete responses combining (Template(if any) + Question, Response)",
    )
    gold_answer: str = Field(..., description="Ground truth answer")
    gold_solution: Optional[str] = Field(
        default="",
        description="Ground truth solution steps, e.g., human-written solutioning.",
    )
    metadata: DataMetadata = Field(
        ...,
        description="Metadata information about the dataset entry. Additional fields are allowed.",
    )
