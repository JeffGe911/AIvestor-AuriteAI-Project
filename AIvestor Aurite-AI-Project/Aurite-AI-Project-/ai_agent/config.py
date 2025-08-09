"""
Configuration management for the AI Macro Analysis Agent.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class APIConfig(BaseModel):
    """API configuration settings."""
    fred_api_key: str = Field(default="", description="FRED API key")
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage API key")
    quandl_api_key: str = Field(default="", description="Quandl API key")
    yahoo_finance_enabled: bool = Field(default=True, description="Enable Yahoo Finance")
    fred_enabled: bool = Field(default=True, description="Enable FRED API")
    alpha_vantage_enabled: bool = Field(default=False, description="Enable Alpha Vantage")
    cache_duration: int = Field(default=3600, description="Cache duration in seconds")
    max_retries: int = Field(default=3, description="Maximum API retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries")


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    url: str = Field(..., description="Complete database URL")
    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    macro_table: str = Field(default="macro indicators", description="Macro data table name")
    predictions_table: str = Field(default="market_predictions", description="Predictions table name")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration settings."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="OpenAI model to use")
    max_tokens: int = Field(default=1000, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="Response creativity level")


class ModelConfig(BaseModel):
    """Machine learning model configuration."""
    model_path: str = Field(default="models/enhanced_nasdaq_model.pkl", description="Path to saved model")
    scaler_path: str = Field(default="models/feature_scaler.pkl", description="Path to saved scaler")
    feature_columns_path: str = Field(default="models/feature_columns.json", description="Path to feature columns")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence for predictions")


class AgentConfig(BaseModel):
    """General agent configuration."""
    name: str = Field(default="MacroAnalysisAgent", description="Agent name")
    log_level: str = Field(default="INFO", description="Logging level")
    batch_size: int = Field(default=100, description="Batch size for processing")


class Config:
    """Main configuration class for the AI agent."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # API Configuration (replaces database)
        self.api = APIConfig(
            fred_api_key=self._get_env("FRED_API_KEY", ""),
            alpha_vantage_api_key=self._get_env("ALPHA_VANTAGE_API_KEY", ""),
            quandl_api_key=self._get_env("QUANDL_API_KEY", ""),
            yahoo_finance_enabled=self._get_env("YAHOO_FINANCE_ENABLED", "true").lower() == "true",
            fred_enabled=self._get_env("FRED_ENABLED", "true").lower() == "true",
            alpha_vantage_enabled=self._get_env("ALPHA_VANTAGE_ENABLED", "false").lower() == "true",
            cache_duration=int(self._get_env("API_CACHE_DURATION", "3600")),
            max_retries=int(self._get_env("API_MAX_RETRIES", "3")),
            retry_delay=float(self._get_env("API_RETRY_DELAY", "1.0"))
        )
        
        # Database Configuration (kept for backward compatibility, but not used)
        db_url = None
        for env_var in ["NEON_DATABASE_URL", "NEON_DB_URL", "DATABASE_URL"]:
            db_url = os.getenv(env_var)
            if db_url:
                break
        
        if db_url:
            import urllib.parse
            parsed = urllib.parse.urlparse(db_url)
            
            self.database = DatabaseConfig(
                url=db_url,
                host=parsed.hostname or self._get_env("NEON_DB_HOST", "localhost"),
                port=parsed.port or int(self._get_env("NEON_DB_PORT", "5432")),
                name=parsed.path.lstrip('/') or self._get_env("NEON_DB_NAME", "postgres"),
                user=parsed.username or self._get_env("NEON_DB_USER", "postgres"),
                password=parsed.password or self._get_env("NEON_DB_PASSWORD", ""),
                macro_table=self._get_env("MACRO_TABLE_NAME", "macro_indicators"),
                predictions_table=self._get_env("PREDICTIONS_TABLE_NAME", "market_predictions")
            )
        else:
            # Create a dummy database config for backward compatibility
            self.database = DatabaseConfig(
                url="dummy://localhost",
                host="localhost",
                name="dummy",
                user="dummy",
                password="dummy",
                macro_table="dummy",
                predictions_table="dummy"
            )
        
        # OpenAI configuration (optional)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai = OpenAIConfig(
            api_key=openai_api_key,
            model=self._get_env("OPENAI_MODEL", "gpt-4o"),
            max_tokens=int(self._get_env("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(self._get_env("OPENAI_TEMPERATURE", "0.7"))
        )
        
        self.model = ModelConfig(
            model_path=self._get_env("MODEL_PATH", "models/enhanced_nasdaq_model.pkl"),
            scaler_path=self._get_env("SCALER_PATH", "models/feature_scaler.pkl"),
            feature_columns_path=self._get_env("FEATURE_COLUMNS_PATH", "models/feature_columns.json"),
            confidence_threshold=float(self._get_env("PREDICTION_CONFIDENCE_THRESHOLD", "0.6"))
        )
        
        self.agent = AgentConfig(
            name=self._get_env("AGENT_NAME", "MacroAnalysisAgent"),
            log_level=self._get_env("LOG_LEVEL", "INFO"),
            batch_size=int(self._get_env("BATCH_SIZE", "100"))
        )
    
    def _get_env(self, key: str, default: Optional[str] = None) -> str:
        """Get environment variable with default."""
        return os.getenv(key, default) if default is not None else os.getenv(key, "")
    
    def validate(self) -> bool:
        """Validate configuration."""
        try:
            # Check if at least one API is enabled
            if not (self.api.yahoo_finance_enabled or self.api.fred_enabled or self.api.alpha_vantage_enabled):
                raise ValueError("At least one API must be enabled")
            
            # Check if required API keys are provided
            if self.api.fred_enabled and not self.api.fred_api_key:
                raise ValueError("FRED API key required when FRED is enabled")
            
            if self.api.alpha_vantage_enabled and not self.api.alpha_vantage_api_key:
                raise ValueError("Alpha Vantage API key required when Alpha Vantage is enabled")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(api_enabled={self.api.yahoo_finance_enabled or self.api.fred_enabled}, openai_enabled={bool(self.openai.api_key)})" 