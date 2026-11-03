import pytest
from rba.config import Config
from rba.errors import RBAError
from rba.llm.prompt import language_instruction


def test_language_default_zh_and_validation():
    cfg = Config()
    assert cfg.language == "zh"
    assert "中文" in language_instruction("zh")
    assert "English" in language_instruction("en")
    cfg.language = "fr"
    with pytest.raises(RBAError):
        cfg.validate_static()
