import time
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from starlette.websockets import WebSocketDisconnect
from torch.nn.functional import threshold

from new_main import IntegratedQASystem

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float

qa_system = IntegratedQASystem()
# todo 1. FastAPI应用实例初始化 和 全局中间件配置
app = FastAPI(
    title="问答系统API",
    description="集成MySQL和RAG的智能问答系统"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 创建静态资源存储目录 static
os.makedirs("static", exist_ok=True) # exist_ok=True目录已存在是不抛出异常.
# 静态资源挂载与页面路由配置
app.mount("/static", StaticFiles(directory="static"), name="static")

GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是黑马程序员，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"
    }
]
def check_greeting(query:str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# todo Http Restful业务接口定义
@app.post('/api/create_session')
async def create_session():
    session_id = str(uuid.uuid4())
    return {'session_id': session_id}

@app.get('/api/history/{session_id}')
async def get_history(session_id:str):
    try:
        history = qa_system.get_session_history(session_id)
        return {
            'session_id': session_id,
            'history': history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

@app.delete('/api/history/{session_id}')
async def clear_history(session_id:str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# 获取有效学科类别接口
@app.get("/api/sources")
async def get_sources():
    return {
        'sources': qa_system.config.VALID_SOURCES
    }

print(f'❤️ * 10 {qa_system.config.VALID_SOURCES}')

# todo 健康检查接口 -> 运维监控接口
# k8s: 管理和部署容器， 基于docker(其他虚拟化框架)一个管理工具
# 探针机制(去调用接口知道当前的服务它的状态)： 就绪探针(探测是不是启动)、 存活探针(是不是正常服务)
@app.get("/health")
async def health_check():
    # 返回健康状态标记, k8s调用该接口返回200则判断服务正常运行.
    return {"status": "healthy"}  # 返回健康状态

@app.post("/api/query")
async def query(request:QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 返回问候回复
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # BM25
    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        return {
            'answer': '请使用WebSocket接口获取流式响应',
            'is_streaming': True,
            'session_id': session_id,
            'processing_time': time.time() - start_time,
        }

    return {'answer': answer, 'is_streaming': False, 'session_id': session_id, "processing_time": time.time() - start_time}


# todo WebSocket接口
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query = request_data.get('query')
            source_filter = request_data.get('source_filter')
            session_id = request_data.get('session_id', str(uuid.uuid4()))
            start_time = time.time()
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": 'start',
                    "session_id": session_id,
                })
            # 日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": 'token',
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break

            # 调用问答系统,流式处理
            collected_answer=''
            for token, is_complete in qa_system.query(query, source_filter, session_id):
                collected_answer += token
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break

                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })


                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break

                await asyncio.sleep(0.01)

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            print(f"Error closing WebSocket: {str(e)}")

if __name__ == '__main__':
    import os
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    uvicorn.run("app:app", host=host, port=port, reload=False)
