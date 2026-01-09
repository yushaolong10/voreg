# server.py
"""
主入口文件，注册所有路由
"""
from fastapi import FastAPI

from router.voice import router as voice_router
from router.eou import router as eou_router


app = FastAPI(
    title="语音服务API", version="1.0.0"
)

# 注册路由
app.include_router(voice_router)
app.include_router(eou_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
