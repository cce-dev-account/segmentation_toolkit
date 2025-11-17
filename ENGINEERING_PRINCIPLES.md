Engineering Principles for Claude CodeOverview
This document defines engineering principles and coding standards for working with Claude Code to produce robust, maintainable, and production-ready code. Following these principles will help avoid common brittleness patterns and ensure code quality.Core Principles1. Defensive Programming First

Validate all inputs at function boundaries
Explicit error handling with specific exception types
Guard clauses early - fail fast with clear messages
Type hints on everything - catch issues before runtime
Null/None checks - never assume values exist
2. Composition Over Heavy Classes
Keep classes focused on data representation. Move business logic to standalone functions.python# ❌ BAD: Heavy class with too many responsibilities
class DocumentProcessor:
    def read_file(self): ...
    def parse_content(self): ...
    def validate_schema(self): ...
    def transform_data(self): ...
    def save_results(self): ...

# ✅ GOOD: Focused classes with standalone functions
class Document:
    """Just holds document state"""
    content: str
    metadata: DocumentMetadata

# Standalone, testable functions
def read_document(path: Path) -> Document: ...
def parse_document(doc: Document) -> ParsedDocument: ...3. Granular Single-Purpose Functions

Maximum 20-30 lines per function
Each function does ONE thing
Name clearly describes what it does
Easy to test in isolation
4. Type-Safe Data Structures with Pydantic
Replace dictionaries with validated Pydantic models:python# ❌ BAD: Dictionary with no structure
def process(params: dict) -> dict:
    threshold = params.get('threshold')  # Could be anything

# ✅ GOOD: Validated Pydantic model
from pydantic import BaseModel, Field
from typing import Literal

class ProcessingParams(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)
    method: Literal['kmeans', 'hierarchical', 'dbscan']
    max_iterations: int = Field(gt=0, default=100)5. No Hardcoded Values or Paths
Everything must be configurable and environment-agnostic:python# ❌ BAD: Hardcoded paths and values
class Processor:
    def load(self):
        df = pd.read_csv('/Users/john/data/segments_2024.csv')
        self.threshold = 0.75  # Magic number
        self.api_key = 'sk-abc123'  # Hardcoded credential
    
    def process(self):
        if segment.type == 'IRB_MODEL_V3':  # Specific to one dataset
            return self.process_irb_v3()

# ✅ GOOD: Configurable and generic
from pathlib import Path
from pydantic import BaseModel, SecretStr

class ProcessorConfig(BaseModel):
    data_path: Path
    threshold: float = Field(ge=0.0, le=1.0)
    api_key: SecretStr
    segment_type_mapping: Dict[str, str] = {}

class Processor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
    
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.config.data_path)
    
    def process(self, segment: Segment) -> Result:
        handler = self.config.segment_type_mapping.get(
            segment.type, 
            self.default_handler
        )
        return handler(segment)6. Demo-Driven Development
Every module must include runnable demos that show how to use it:python# ❌ BAD: No examples or only production code

# ✅ GOOD: Include demo.py or __main__ block
"""
segment_processor/
├── __init__.py
├── processor.py
├── models.py
├── demo.py          # Runnable demonstration
└── tests/
    ├── test_processor.py
    └── fixtures.py  # Test data generators
"""

# demo.py
def demo_basic_usage():
    """Demonstrate basic segment processing."""
    # Create sample data
    sample_segments = [
        Segment(id="demo-1", type="behavioral", score=0.8),
        Segment(id="demo-2", type="demographic", score=0.6),
    ]
    
    # Configure processor
    config = ProcessingParams(
        threshold=0.7,
        method="kmeans",
        min_size=10
    )
    
    # Process
    processor = SegmentProcessor(config)
    results = processor.process_all(sample_segments)
    
    # Display results
    print(f"Processed {len(results)} segments successfully")
    for result in results:
        print(f"  - {result.id}: {result.status}")
    
    return results

def demo_error_handling():
    """Demonstrate error handling."""
    try:
        # Intentionally invalid config to show validation
        config = ProcessingParams(threshold=1.5)  # Will raise ValidationError
    except ValidationError as e:
        print(f"Configuration validation caught: {e}")
    
if __name__ == "__main__":
    print("=== Basic Usage Demo ===")
    demo_basic_usage()
    
    print("\n=== Error Handling Demo ===")
    demo_error_handling()7. Test Data Generators Instead of Fixed Files
Never reference specific test files. Generate test data programmatically:python# ❌ BAD: Hardcoded test files
def test_processor():
    df = pd.read_csv('test_data/segments_final_v2.csv')
    processor.process(df)

# ✅ GOOD: Test data generators
from typing import List
import random
from faker import Faker

fake = Faker()

class TestDataGenerator:
    """Generate test data on demand."""
    
    @staticmethod
    def make_segment(
        score: Optional[float] = None,
        size: Optional[int] = None,
        type: Optional[str] = None
    ) -> Segment:
        """Create a single test segment with optional overrides."""
        return Segment(
            id=f"test-{fake.uuid4()}",
            type=type or random.choice(['behavioral', 'demographic']),
            score=score if score is not None else random.random(),
            size=size or random.randint(1, 1000)
        )
    
    @staticmethod
    def make_segments(
        count: int = 10,
        **kwargs
    ) -> List[Segment]:
        """Create multiple test segments."""
        return [TestDataGenerator.make_segment(**kwargs) for _ in range(count)]
    
    @staticmethod
    def make_edge_cases() -> Dict[str, Segment]:
        """Generate edge case segments for testing."""
        return {
            'empty': Segment(id='empty', type='test', score=0.0, size=0),
            'minimal': Segment(id='min', type='test', score=0.001, size=1),
            'maximal': Segment(id='max', type='test', score=0.999, size=999999),
            'boundary': Segment(id='boundary', type='test', score=0.5, size=100),
        }

# In tests
def test_processor_with_generated_data():
    # Generate test data
    segments = TestDataGenerator.make_segments(
        count=50,
        score=0.8  # All with high scores
    )
    
    processor = SegmentProcessor(test_config)
    results = processor.process_all(segments)
    
    assert len(results) == 508. Clean Code Structure

All imports at the top of the file - never inside functions
No nested function definitions - use private helpers instead
No lambda functions except for simple key functions in sort/max/min
Consistent snake_case naming for variables and functions
PascalCase for classes only
9. Abstract External Dependencies
Use dependency injection and interfaces for all external systems:python# ❌ BAD: Direct coupling to external systems
def send_notification(message: str):
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('hardcoded@email.com', 'password123')
    server.send_message(message)

# ✅ GOOD: Abstract interface
from abc import ABC, abstractmethod

class NotificationService(ABC):
    @abstractmethod
    def send(self, message: str) -> bool:
        pass

class EmailNotificationService(NotificationService):
    def __init__(self, config: EmailConfig):
        self.config = config
    
    def send(self, message: str) -> bool:
        # Implementation details
        ...

class MockNotificationService(NotificationService):
    """For testing."""
    def __init__(self):
        self.sent_messages = []
    
    def send(self, message: str) -> bool:
        self.sent_messages.append(message)
        return True

# Usage
def process_with_notification(
    data: Data,
    notifier: NotificationService  # Injected dependency
) -> Result:
    result = process(data)
    notifier.send(f"Processing complete: {result.id}")
    return result10. Precise Exception Handling
python# ❌ BAD: Broad catch-all
try:
    # 100 lines of code
    process_everything()
except Exception as e:
    print(f"Failed: {e}")

# ✅ GOOD: Specific handling
try:
    data = load_data()
except FileNotFoundError as e:
    raise DataLoadError(f"Input file not found: {e}")

try:
    validated = validate(data)
except ValidationError as e:
    raise DataValidationError(f"Invalid data format: {e}")11. Immutable Data Structures by Default
python# ✅ GOOD: Immutable with frozen dataclasses
from dataclasses import dataclass, replace
from typing import Tuple

@dataclass(frozen=True)
class Segment:
    items: Tuple[str, ...]  # Immutable tuple
    score: float
    
    def with_item(self, item: str) -> 'Segment':
        """Return new segment with added item."""
        return replace(self, items=self.items + (item,))12. Explicit Constraints with Enums
pythonfrom enum import Enum
from typing import Literal

class ProcessingMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"

class Config(BaseModel):
    mode: ProcessingMode  # Not just 'str'
    level: Literal["low", "medium", "high"]Critical RulesNEVER Use Unicode/Emoji in Code

NO emoji in comments ❌
NO unicode symbols in strings (except for legitimate text data)
NO decorative characters in docstrings or print statements
Use ASCII only for all code, comments, and debugging output
This includes: ✅ ❌ 🎉 📝 🔍 → ⚠️ 💡 or any other unicode symbols
NEVER Hardcode

NO hardcoded file paths - use Path objects and configuration
NO hardcoded URLs - use environment variables or config files
NO dataset-specific logic - use configuration mappings
NO magic numbers - use named constants
NO specific test file references - use generators
NO environment-specific code - use abstraction layers
String Literals

Use single quotes for short strings and dictionary keys: 'key'
Use double quotes for messages and docstrings: "This is a message"
Use raw strings for regex patterns: r'\d+\.txt'
No f-strings with complex expressions - extract to variables first
Complete Type Hints
Every function must have complete type hints:
pythonfrom typing import List, Optional, Dict, Tuple

def process_segments(
    data: List[Segment],
    params: ProcessingParams,
    validate: bool = True
) -> Tuple[List[Result], Metrics]:
    ...Testing RequirementsTest Structure
Every module must include:
module/
├── __init__.py
├── core.py           # Core implementation
├── models.py         # Data models
├── demo.py          # Runnable demonstrations
└── tests/
    ├── __init__.py
    ├── test_core.py
    ├── test_models.py
    └── fixtures.py  # Test data generatorsTest Patterns
python# ✅ GOOD: AAA Pattern with generated data
def test_segment_processing():
    # Arrange
    segments = TestDataGenerator.make_segments(count=10)
    processor = Processor(TestConfigFactory.default())
    
    # Act
    results = processor.process_all(segments)
    
    # Assert
    assert len(results) == 10
    assert all(r.status == "success" for r in results)

# ✅ GOOD: Parametrized tests
@pytest.mark.parametrize("score,expected", [
    (0.0, False),
    (0.5, True),
    (1.0, True),
    (-0.1, ValueError),
])
def test_validation_thresholds(score, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            Segment(score=score)
    else:
        segment = Segment(score=score)
        assert segment.is_valid() == expectedTest Data Builders
pythonclass SegmentBuilder:
    """Fluent builder for test segments."""
    
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.id = f"test-{uuid.uuid4()}"
        self.score = 0.5
        self.items = []
        return self
    
    def with_score(self, score: float) -> 'SegmentBuilder':
        self.score = score
        return self
    
    def with_items(self, *items) -> 'SegmentBuilder':
        self.items.extend(items)
        return self
    
    def build(self) -> Segment:
        segment = Segment(
            id=self.id, 
            score=self.score, 
            items=tuple(self.items)
        )
        self._reset()
        return segmentCommon Anti-Patterns to Avoid
Dictionary abuse - Use Pydantic models instead
String typing - Use proper imports, not 'ClassName'
Hidden dependencies - All imports at the top
Hardcoded anything - Paths, URLs, credentials, dataset names
Dataset-specific logic - Use configuration and abstraction
Mutable default arguments - Use None and create inside function
Catch-all exception handlers - Be specific about what you catch
Global variables - Pass configuration explicitly
Side effects in property getters - Properties should only return values
Print debugging - Use proper logging instead
Fixed test data files - Use test data generators
Nested function definitions - Use module-level private functions
Mixed sync/async - Keep them separate and clearly named
God objects - Break into smaller, focused components
Project Structure Templateyour_project/
├── README.md
├── ENGINEERING_PRINCIPLES.md
├── requirements.txt
├── setup.py
├── .env.example           # Example environment variables
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration management
├── src/
│   └── your_module/
│       ├── __init__.py
│       ├── core.py       # Core logic
│       ├── models.py     # Pydantic models
│       ├── interfaces.py # Abstract interfaces
│       ├── demo.py       # Runnable examples
│       └── utils.py      # Helper functions
├── tests/
│   ├── conftest.py       # Pytest configuration
│   ├── fixtures.py       # Test data generators
│   ├── test_core.py
│   └── test_models.py
└── examples/
    ├── basic_usage.py    # Simple example
    └── advanced_usage.py # Complex exampleCode Review ChecklistBefore accepting code from Claude Code, verify:
 No emoji or unicode symbols in code or comments
 No hardcoded paths, URLs, or dataset-specific logic
 All functions have complete type hints
 Data structures use Pydantic models, not raw dictionaries
 No nested function definitions
 Functions are granular (max 30 lines)
 Error handling is specific, not broad try/except blocks
 All imports are at the top of the file
 String values use Enums or Literals where appropriate
 Classes are lightweight with most logic in standalone functions
 Variable names use snake_case consistently
 Docstrings for all public functions
 Demo code showing how to use the module
 Test data generators instead of fixed test files
 No lambda functions except for simple sort keys
 External dependencies are abstracted with interfaces
 Configuration uses Pydantic models with validation
Template for Claude Code RequestsWhen requesting code from Claude Code, use this template:Create [component description] following these requirements:

Core Requirements:
- Use Pydantic BaseModel for all data structures (no raw dictionaries)
- Write granular functions (max 20-30 lines each)
- Include complete type hints for all functions
- Use dependency injection for external dependencies
- No nested function definitions
- Handle errors with specific exceptions, not broad try/except
- Use Enums or Literal types for constrained string values
- Follow snake_case naming for variables and functions
- Keep classes minimal - prefer standalone functions

Critical Rules:
- NO emoji or unicode symbols anywhere in the code
- NO hardcoded paths, URLs, or dataset-specific logic
- NO imports inside functions
- NO dictionary string keys for structured data - use Pydantic
- NO lambda functions except for sort/max/min keys
- NO references to specific data files - use test generators
- ASCII characters only in all code and comments

Include:
- Runnable demo.py showing usage examples
- Test data generators for creating sample data
- Abstract interfaces for external dependencies
- Configuration through Pydantic models, not hardcoded values

Structure:
- All imports at the top, grouped by: stdlib, third-party, local
- Each function does ONE thing with a clear name
- Validate inputs at function boundaries
- Return structured types (Pydantic models), not dictionaries