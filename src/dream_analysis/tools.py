"""Public compatibility facade for dream-agent tools.

Implementations live in focused modules; imports from dream_analysis.tools
remain supported.
"""

from dream_analysis.analytics_tools import DreamStatisticsTool, TagTrendTool
from dream_analysis.character_tools import (
    CharacterContextTool,
    CharacterMentionsTool,
)
from dream_analysis.journal_tools import (
    DreamByIdTool,
    DreamDateRangeTool,
    DreamTagTool,
)
from dream_analysis.retrieval_tools import DreamKeywordSearchTool, DreamSearchTool
from dream_analysis.tool_protocols import (
    AgentTool,
    AnalyticalToolResult,
    CharacterRecordRepository,
    DatedDreamRepository,
    DreamByIdRepository,
    DreamCollectionRepository,
    SearchableDreamIndex,
    SearchableDreamKeywordIndex,
    StructuredDreamRecordRepository,
    TaggedDreamRepository,
)

__all__ = [
    "AgentTool",
    "AnalyticalToolResult",
    "CharacterContextTool",
    "CharacterMentionsTool",
    "CharacterRecordRepository",
    "DatedDreamRepository",
    "DreamByIdRepository",
    "DreamByIdTool",
    "DreamCollectionRepository",
    "DreamDateRangeTool",
    "DreamKeywordSearchTool",
    "DreamSearchTool",
    "DreamStatisticsTool",
    "DreamTagTool",
    "SearchableDreamIndex",
    "SearchableDreamKeywordIndex",
    "StructuredDreamRecordRepository",
    "TaggedDreamRepository",
    "TagTrendTool",
]
