"""
Exception hierarchy for IBD Crew system.

This module defines all custom exceptions used throughout the IBD Crew
agents and tools. Using specific exceptions makes error handling more
robust and enables better error recovery strategies.

Reference: IBD Momentum Investment Framework v4.0 - Error Handling (§10.2)
"""

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorSeverity(Enum):
    """Categorizes the severity of errors for handling decisions."""
    
    CRITICAL = "critical"
    """Pipeline should stop; this error prevents meaningful continuation"""
    
    WARNING = "warning"
    """Log but continue; we can proceed despite this issue"""
    
    INFO = "info"
    """Track but don't alarm; this is informational"""


@dataclass
class ProcessingError:
    """
    Structured error record for tracking failures during file/data processing.
    
    Used to collect all errors in a pipeline run so they can be logged,
    analyzed, and reported in output without stopping execution (for WARNING/INFO severity).
    """
    
    file_name: str
    """Name of the file that caused the error"""
    
    error_type: str
    """Category of error (e.g., "PDF_PARSE_ERROR", "VALIDATION_ERROR")"""
    
    message: str
    """Human-readable error message"""
    
    severity: ErrorSeverity
    """How serious is this error? CRITICAL/WARNING/INFO"""
    
    traceback_str: Optional[str] = None
    """Full traceback for debugging (only for CRITICAL/WARNING)"""
    
    context: dict = field(default_factory=dict)
    """Additional context data (symbol, row number, etc.)"""
    
    @classmethod
    def from_exception(
        cls,
        file_name: str,
        error_type: str,
        exception: Exception,
        severity: ErrorSeverity,
        context: Optional[dict] = None,
    ) -> "ProcessingError":
        """
        Create ProcessingError from a caught exception.
        
        Args:
            file_name: Name of file being processed
            error_type: Custom error category
            exception: The exception that was caught
            severity: How to categorize this error
            context: Optional additional context data
        
        Returns:
            ProcessingError with traceback automatically extracted
        """
        tb_str = traceback.format_exc() if severity in (ErrorSeverity.CRITICAL, ErrorSeverity.WARNING) else None
        return cls(
            file_name=file_name,
            error_type=error_type,
            message=str(exception),
            severity=severity,
            traceback_str=tb_str,
            context=context or {},
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "file_name": self.file_name,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "traceback": self.traceback_str,
            "context": self.context,
        }


# ============================================================================
# BASE EXCEPTION CLASSES
# ============================================================================

class IBDCrewException(Exception):
    """
    Base exception for all IBD Crew errors.
    
    Inheriting from this allows catching all framework errors:
        try:
            ...
        except IBDCrewException as e:
            ...
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__


class DataProcessingError(IBDCrewException):
    """Base class for errors during data extraction and processing."""
    pass


class ValidationError(IBDCrewException):
    """Base class for data validation failures."""
    pass


class ConfigurationError(IBDCrewException):
    """Base class for configuration/setup issues."""
    pass


class PipelineError(IBDCrewException):
    """Base class for agent pipeline execution errors."""
    pass


# ============================================================================
# FILE READING EXCEPTIONS (§2 - Research Agent)
# ============================================================================

class FileReadError(DataProcessingError):
    """
    Raised when a file cannot be read or opened.
    
    Example:
        raise FileReadError("Unable to open IBD_50.xls: Permission denied")
    """
    pass


class PDFParseError(DataProcessingError):
    """
    Raised when PDF parsing fails.
    
    This includes pdfplumber errors, missing pages, unreadable content, etc.
    
    Example:
        raise PDFParseError("Cannot extract text from page 2 of IBD_SmartList.pdf")
    """
    pass


class ExcelParseError(DataProcessingError):
    """
    Raised when Excel/CSV parsing fails.
    
    This includes xlrd errors, missing sheets, corrupted data, etc.
    
    Example:
        raise ExcelParseError("Sheet 'IBD 50' not found in IBD_Leaders.xls")
    """
    pass


class MorningstarParseError(DataProcessingError):
    """
    Raised when Morningstar PDF parsing fails.
    
    Example:
        raise MorningstarParseError("Unable to extract ratings from M* PDF")
    """
    pass


# ============================================================================
# DATA VALIDATION EXCEPTIONS (§3 - Analyst Agent)
# ============================================================================

class DataValidationError(ValidationError):
    """
    Raised when extracted data doesn't match expected schema.
    
    This is a CRITICAL error that should stop the pipeline.
    
    Example:
        raise DataValidationError("Stock record missing required 'symbol' field")
    """
    pass


class RatingValidationError(ValidationError):
    """
    Raised when IBD ratings are invalid or out of range.
    
    Example:
        raise RatingValidationError("Composite rating 150 is out of range [0, 99]")
    """
    pass


class SectorValidationError(ValidationError):
    """
    Raised when sector classification is invalid.
    
    Example:
        raise SectorValidationError("Unknown sector 'XYZ'; not in IBD sectors list")
    """
    pass


class SchemaValidationError(ValidationError):
    """
    Raised when Pydantic schema validation fails.
    
    This wraps Pydantic ValidationError for consistency.
    
    Example:
        raise SchemaValidationError("ResearchStock validation failed: ...")
    """
    pass


# ============================================================================
# CLASSIFICATION EXCEPTIONS (§4 - Agents 03-05)
# ============================================================================

class ClassificationError(DataProcessingError):
    """Base class for sector/cap/tier classification failures."""
    pass


class SectorClassificationError(ClassificationError):
    """
    Raised when sector classification fails.
    
    Example:
        raise SectorClassificationError("LLM unable to classify symbol 'XYZ' into a sector")
    """
    pass


class CapClassificationError(ClassificationError):
    """
    Raised when market cap classification fails.
    
    Example:
        raise CapClassificationError("Unable to determine market cap for symbol 'XYZ'")
    """
    pass


class TierAssignmentError(ClassificationError):
    """
    Raised when tier assignment logic fails.
    
    Example:
        raise TierAssignmentError("Cannot assign tier to stock with missing ratings")
    """
    pass


# ============================================================================
# EVALUATION & CALCULATION EXCEPTIONS (§5 - All Agents)
# ============================================================================

class CalculationError(DataProcessingError):
    """
    Raised when mathematical calculations fail unexpectedly.
    
    Example:
        raise CalculationError("Division by zero in Sharpe ratio calculation")
    """
    pass


class MetricsCalculationError(CalculationError):
    """
    Raised when risk/return metrics cannot be calculated.
    
    Example:
        raise MetricsCalculationError("Insufficient data to calculate volatility")
    """
    pass


class BacktestError(CalculationError):
    """
    Raised when backtesting or scenario analysis fails.
    
    Example:
        raise BacktestError("Cannot run returns projection: missing portfolio data")
    """
    pass


# ============================================================================
# CONFIGURATION & SETUP EXCEPTIONS (§10 - Framework)
# ============================================================================

class EnvConfigError(ConfigurationError):
    """
    Raised when required environment variables are missing.
    
    Example:
        raise EnvConfigError("Missing required env var: OPENAI_API_KEY")
    """
    pass


class DataDirectoryError(ConfigurationError):
    """
    Raised when data directory structure is invalid.
    
    Example:
        raise DataDirectoryError("Data directory '/data/' not found")
    """
    pass


class LLMConfigError(ConfigurationError):
    """
    Raised when LLM configuration is invalid or unavailable.
    
    Example:
        raise LLMConfigError("Claude API key not configured; LLM classification disabled")
    """
    pass


# ============================================================================
# PIPELINE EXECUTION EXCEPTIONS (§9 - run_pipeline.py)
# ============================================================================

class AgentExecutionError(PipelineError):
    """
    Raised when an agent fails to execute its task.
    
    This is typically a CRITICAL error that stops the pipeline.
    
    Example:
        raise AgentExecutionError("Research Agent failed: unable to process data")
    """
    pass


class DataFlowError(PipelineError):
    """
    Raised when expected output from one agent is missing for the next.
    
    Example:
        raise DataFlowError(
            "Analyst Agent requires ResearchOutput, but Research Agent returned None"
        )
    """
    pass


class OutputWriteError(PipelineError):
    """
    Raised when Excel/CSV output cannot be written.
    
    Example:
        raise OutputWriteError("Cannot write agent02_analyst_2026-02-11.xlsx: Permission denied")
    """
    pass


class IntegrationError(PipelineError):
    """
    Raised when cross-agent integration fails.
    
    Example:
        raise IntegrationError("Rotation detector inconsistent with analyst findings")
    """
    pass


# ============================================================================
# UTILITY FUNCTIONS FOR ERROR HANDLING
# ============================================================================

def wrap_exception_as_processing_error(
    exception: Exception,
    file_name: str,
    error_type: str,
    severity: ErrorSeverity = ErrorSeverity.WARNING,
) -> ProcessingError:
    """
    Convert any exception to ProcessingError for unified logging.
    
    Args:
        exception: The exception to wrap
        file_name: Name of file being processed
        error_type: Custom categorization
        severity: How to treat this error (default: WARNING)
    
    Returns:
        ProcessingError ready for logging
    """
    return ProcessingError.from_exception(
        file_name=file_name,
        error_type=error_type,
        exception=exception,
        severity=severity,
    )
