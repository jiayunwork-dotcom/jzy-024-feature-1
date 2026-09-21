"""业务错误类型。

HTTP 层统一把 ServiceError 翻成带中文原因的 4xx 响应；
各计算/存储模块只抛这个异常，不接触 FastAPI。
"""


class ServiceError(Exception):
    """所有可预期的业务错误（参数非法、对象档不存在等）。"""

    def __init__(self, reason: str, status_code: int = 400):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
