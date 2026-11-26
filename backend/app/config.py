from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "ReturnIQ API"
    frontend_origin: str = "http://localhost:5173"
    database_url: str = "postgresql+psycopg://postgres:CHANGE_ME@localhost:5432/returniq"
    model_dir: str = "models/tfidf"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
