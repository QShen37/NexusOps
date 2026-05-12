"""
AIOps 请求和响应模型
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AIOpsRequest(BaseModel):
    """AIOps 诊断请求"""
    session_id: Optional[str] = Field(
        default="default",
        description="会话ID，用于追踪诊断历史"
    )
    user_input: Optional[str] = Field(
        default=None,
        description="自然语言任务描述；为空时由服务端使用默认巡检提示"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-123",
                "user_input": "请检查 CPU、内存与磁盘使用情况"
            }
        }

class AlertInfo(BaseModel):
    """告警信息"""
    alertname: str
    severity: str
    instance: str
    duration: str
    description: Optional[str] = None


class DiagnosisResponse(BaseModel):
    """诊断响应（非流式）"""

    code: int = 200
    message: str = "success"
    data: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "completed",
                    "target_alert": {
                        "alertname": "HighCPUUsage",
                        "severity": "critical"
                    },
                    "diagnosis": {
                        "root_cause": "数据库连接池耗尽",
                        "recommendations": ["扩容数据库连接池", "优化SQL查询"]
                    }
                }
            }
        }
