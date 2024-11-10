from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    This class is used to load environment variables
    from the specified `.env.local` file.
    """

    BOT_TOKEN: str
    GEMINI_API_KEY: str
    ALLOWED_CHATS: list[int]

    class Config:  # pylint: disable=R0903
        """
        Configuration class that specifies the location of
        the .env file to be used for loading environment variables.
        """
        env_file = "./.env.local"


# Load the settings from the .env file
settings = Settings()