from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NFLMI_", env_file=".env", extra="ignore")
    database_path: str = "data/nfl_market.duckdb"
    mcp_transport: str = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
settings = Settings()
