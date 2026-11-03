"""User-facing errors. Messages are Chinese and always carry a fix hint."""


class RBAError(Exception):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n建议：{self.hint}"
        return self.message
